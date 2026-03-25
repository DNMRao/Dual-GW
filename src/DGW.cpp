#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <complex>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <chrono>

#define NCA_SOLVER_ASSERT_0 0
#define NCA_SOLVER_ASSERT_1 0

#define CNTR_USE_OMP 1
#define CNTR_USE_MPI 1

#include <cntr/cntr.hpp>
#include <cntr/utils/read_inputfile.hpp>
#include <cntr/hdf5/hdf5_interface.hpp>
#include <cntr/cntr_dyson_omp_decl.hpp>
#include <cntr/cntr_dyson_omp_impl.hpp>

#include "formats.hpp"
#include "Dgw_lattice.hpp"
#include "Dgw_kpoint_dec.hpp"
#include "Dgw_kpoint_impl.hpp"
#include "Dgw_selfconsistency_decl.hpp"
#include "Dgw_selfconsistency_impl.hpp"
//#include "Dgw_selfconsistency_impl.hpp"

#include "./ppsc/ppsc.hpp"
#include "./ppsc/solver.hpp"
#include "./ppsc/hilbert_spaces/single_band_fermi_diag.hpp"
#include "./ppsc/hamiltonians/single_band_hubbard.hpp"
#include "ohmic_bath.hpp"
//#include "./ppsc/baths/non_int_boson_propagator.hpp"

#define CPLX std::complex<double>
#define CFUNC cntr::function<double>
#define GREEN cntr::herm_matrix<double>
#define DIST_TIMESTEP cntr::distributed_timestep_array<double>
using namespace std;

// -----------------------------------------------------------------------

typedef ppsc::operator_type operator_type;
typedef ppsc::mam::dynamic_matrix_type matrix_type; 

// -----------------------------------------------------------------------

template<class HILB>
ppsc::pp_ints_type get_pp_ints(ppsc::gf_type & Delta_up, ppsc::gf_type & Delta_up_cc, 
                                HILB & h) {

  // spin (u)p/(d)own and (c)reation/(a)nihilation operators


  int fermion=-1, fwd=+1, bwd=-1;

  ppsc::pp_ints_type pp_ints;

  pp_ints.push_back(ppsc::pp_int_type(Delta_up,0,0,h.cuc,h.cua,fermion,fwd)); // spin up fwd
  pp_ints.push_back(ppsc::pp_int_type(Delta_up_cc,0,0,h.cua,h.cuc,fermion,bwd)); // spin up bwd

  pp_ints.push_back(ppsc::pp_int_type(Delta_up,0,0,h.cdc,h.cda,fermion,fwd)); // spin do fwd // assuming spin sym
  pp_ints.push_back(ppsc::pp_int_type(Delta_up_cc,0,0,h.cda,h.cdc,fermion,bwd)); // spin do bwd // assuming spin sym

  return pp_ints;
}

// -----------------------------------------------------------------------
template<class HILB>
ppsc::gf_verts_type get_gf_verts(HILB & h) {

  ppsc::gf_verts_type gf_verts;
  gf_verts.push_back(ppsc::gf_vert_type(0,0,h.cua,h.cuc)); // spin up

  ppsc::operator_type nbar_c = h.n - h.Q; // 
  gf_verts.push_back(ppsc::gf_vert_type(0, 0, nbar_c, nbar_c)); // charge fluctuation susceptibilty

  ppsc::operator_type n_z = h.m ; // spin-spin susceptibilty
  gf_verts.push_back(ppsc::gf_vert_type(0, 0, n_z, n_z)); // 

//  gf_verts.push_back(ppsc::gf_vert_type(0,0,h.cda,h.cdc)); // spin down
  return gf_verts;
}

//*****************************************************

void efield_to_afield(int nt,double h,std::vector<double> &efield,std::vector<double> &afield,int kt){
        int kt1=(nt>=kt ? kt : nt),n,n1,tstp;
        double At;
        std::vector<double> a_field;
        a_field.resize(nt+2);
        for(tstp=-1;tstp<=nt;tstp++){
//                a_field[tstp+1]=-0.0;
                afield[tstp+1]=0.0;
          }
          for(tstp=0;tstp<=nt;tstp++){
                At=0.0;
                n1=(tstp<kt1 ? kt1 : tstp);
                for(n=0;n<=n1;n++){
                        At += integration::I<double>(kt1).gregory_weights(tstp,n)*efield[n+1];
                }
                afield[tstp+1]=At*(-h);
//                afield[tstp+1]=a_field[tstp+1];
        }
      
   }

//****************************************************************************

class bethedos_2 {  // Ohmic DOS
  public:
    double hi_;
    double lo_;
        bethedos_2(){ lo_=0.01;hi_=15;}
    double operator()(double x){
      double arg=(x/1.0)*exp(-x/1.0);
            return ( arg < 0 ? 0.0 : arg/1.0);
        }
  };


class bethedos_3 {  // Ohmic DOS
  public:
    double hi_;
    double lo_;
        bethedos_3(){ lo_=-2;hi_= 2;}
    double operator()(double x){
       double arg = 4.0  - x * x;
        double num =  3.14159265358979323846 * 2;
        return (arg < 0 ? 0.0 : sqrt(arg) / num);
        }
  };


//**************************************************************************************************
std::vector<vector<double>> get_orb_kinetic_energy(ppsc::gf_type & Gloc, ppsc::gf_type & Delta, double beta, double h, int kt)
{
  std::vector<vector<double>> Ekin;

  if(Gloc.nt() == 0)
  {
          return Ekin; // Do not attempt to convolve equilibrium only.
  }
 
  

   ppsc::gf_type tmp(Gloc.nt(), Gloc.ntau(), Gloc.size1(), Gloc.sig());

  ::cntr::convolution(tmp, Gloc, Gloc, Delta, Delta,
                      integration::I<double>(kt), beta, h);

  ppsc::cntr::herm_matrix_matrix_ref<ppsc::mam::dynamic_matrix_type> tmp_ref(tmp);

  Ekin = std::vector<std::vector<double>> (Gloc.size1(), std::vector<double>(Gloc.nt() + 2));

          Ekin[0][0] = -tmp_ref.mat(tmp_ref.ntau() - 1)(0,0).real();


  for(auto idx : ppsc::range(0, Gloc.size1()))
        for( auto t : ppsc::range(0, Gloc.nt()+1) )
                  Ekin[idx][t + 1] = -Gloc.sig() * tmp_ref.les(t, t)(idx,idx).imag();

  return Ekin;
}
/**/
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

int main(int argc, char *argv[]) {
  int MatsMaxIter, itermax, nt, ntau, iter_rtime, kt, tstp, order, nomp, L, iter;
  double beta, h, dmfterr, errmax, mu, err_e, err_b_c, err_b_z, real_mix, Mat_mix;
  double dope, bath_beta;
  int store_gf, store_pp, mix_sigma, bath_flag;
  bool matsubara_converged = false, mix_b = false;
  int ntasks,tid,tid_root;
  std::vector<int> tid_map;
  std::vector<double> Epulse, U_val, Lamda_;

     typedef ppsc::hilbert_spaces::single_band_fermi_diag hilbert_space_type;
     typedef ppsc::hamiltonians::single_band_hubbard<hilbert_space_type> hamiltonian_type;
     typedef ppsc::solver<hamiltonian_type> solver_type;

     hilbert_space_type hilbert_space;
     hilbert_space.init();
     Dgw::selfconsistency_pm<Dgw::lattice_2d> latt_;


 {
      MPI_Init(&argc,&argv);
      MPI_Comm_size(MPI_COMM_WORLD, &ntasks);
      MPI_Comm_rank(MPI_COMM_WORLD, &tid);
      tid_root=0;
  }

 
//   printf("Hello world from rank %d out of %d processors\n", tid,ntasks);

  try {
    // ---------------------------------------------------------------------
    // READ GENERAL INPUT (NOT YET MICROSCOPIC PARAMETERS)
    {
      if (argc < 2)
        throw("COMMAND LINE ARGUMENT MISSING");
      // scan the input file, double underscores to avoids mismatch
      find_param(argv[1], "__nt=", nt);
      find_param(argv[1], "__ntau=", ntau);
      find_param(argv[1], "__beta=", beta);
      find_param(argv[1], "__h=", h);
      find_param(argv[1], "__kt=", kt);
      find_param(argv[1], "__order=", order);
      find_param(argv[1],"__MatsMaxIter=",MatsMaxIter);
      find_param(argv[1],"__errmax=",errmax);
      find_param(argv[1],"__itermax=",itermax);
      find_param(argv[1],"__iter_rtime=",iter_rtime);
      find_param(argv[1],"__L=",L);
      find_param(argv[1],"__mix_mat=",Mat_mix);
      find_param(argv[1],"__mix_real=",real_mix);
      find_param(argv[1],"__mix_sigma=",mix_sigma);
      find_param(argv[1],"__field_symmetry=",latt_.lattice_model_.field_symmetry_);
      find_param_tvector(argv[1], "__U=", U_val, nt);
      find_param(argv[1], "__nomp=", nomp); 
      find_param(argv[1], "__dope=", dope);
      find_param(argv[1],"__bath_beta=",bath_beta);
      find_param(argv[1], "__U0=", U_val[0]);
      find_param_tvector(argv[1], "__Lamda=", Lamda_, nt);
      find_param(argv[1], "__Lamda0=", Lamda_[0]);
      find_param(argv[1], "__Lamda1=", Lamda_[1]);
      find_param(argv[1], "__bath_flag=", bath_flag);
//      find_param_tvector(argv[1],"__Epulse=",Epulse,nt); 
     }
   
//     std::cout << "HI" << std::endl;

//      nomp=omp_get_max_threads();

     solver_type imp(nt, ntau, beta, h, kt, nomp, hilbert_space, order);


     if(tid==tid_root){
      find_param(argv[1], "__mu=", imp.hamiltonian.mu);
      find_param_tvector(argv[1], "__eps_up=", imp.hamiltonian.eps_up, nt);
      find_param_tvector(argv[1], "__eps_do=", imp.hamiltonian.eps_do, nt);
      find_param_tvector(argv[1], "__U=", imp.hamiltonian.U, nt);
      find_param(argv[1], "__U0=", imp.hamiltonian.U[0]);
//      find_param(argv[1], "__U1=", imp.hamiltonian.U[1]);
      imp.update_hamiltonian();
     }

      

  
    
    latt_.set_OMP(-1);
    latt_.lattice_model_.init(L,nt,latt_.lattice_model_.field_symmetry_);   
   
//    double dis;
//    for(tstp=-1;tstp<=nt;tstp++){
//       double dis=latt_.get_average_disper(tstp);
//      if(tid==tid_root) std::cout << dis << std::endl;
//     }

    find_param_tvector(argv[1],"__U=",latt_.lattice_model_.U0_,nt);

    find_param(argv[1], "__U0=", latt_.lattice_model_.U0_[0]);
//    find_param(argv[1], "__U1=", latt_.lattice_model_.U0_[1]);

    find_param_tvector(argv[1],"__V=",latt_.lattice_model_.V_,nt);

    find_param(argv[1],"__V0=",latt_.lattice_model_.V_[0]);
   
//    find_param(argv[1],"__V1=",latt_.lattice_model_.V_[1]);

    find_param_tvector(argv[1],"__E=",latt_.lattice_model_.E_,nt);

    vector<double> At(nt + 2);
    efield_to_afield(nt,h,latt_.lattice_model_.E_,At,kt);
    for(tstp=-1;tstp<=nt;tstp++) latt_.lattice_model_.A_[tstp+1]=At[tstp+1];
    

    latt_.init(nt,ntau,beta,h,kt);

    
    assert(tid==latt_.gk_all_timesteps_.tid());  

     if(tid==tid_root) {
      std::cout << latt_.lattice_model_.nk_ << std::endl;
//      std::cout << latt_.omp_num_ << std::endl;
      std::cout <<  ntasks << std::endl;
      }

     //double dis;
     for(tstp=-1;tstp<=nt;tstp++){
        double dis=latt_.get_average_disper(tstp);
         if(tid==tid_root){
          double dis1 = dis ; 
          imp.hamiltonian.eps_up[tstp+1] = dis1;
          imp.hamiltonian.eps_do[tstp+1] = dis1;
          //std::cout << dis << std::endl;
         }
      }    
    
       if(tid==tid_root) imp.update_hamiltonian();

//        for(tstp=-1;tstp<=nt;tstp++){
//         if(tid==tid_root) std::cout << imp.hamiltonian.eps_up[tstp+1] << imp.hamiltonian.eps_do[tstp+1] << std::endl;
//        }

    if(tid==tid_root){   
     FILE *out; 
     out=fopen("A_pulse.out","w");
     for(tstp=-1;tstp<=nt;tstp++){ 
      fprintf(out,"t: %i ",tstp);
      fprintf(out," E: %.10g",latt_.lattice_model_.E_[tstp+1]);
      fprintf(out," A: %.10g",latt_.lattice_model_.A_[tstp+1]);
//      fprintf(out," U0: %.10g",latt_.lattice_model_.U0_[tstp+1]);
//      fprintf(out," U_val: %.10g",U_val[tstp+1]); 
      fprintf(out,"\n");
     }
      fclose(out);
    }

//   exit(0);

   if(tid==tid_root){
      print_line_minus(50);
      std::cout << " Estimation of memory requirements" << std::endl;
      print_line_minus(50);

      const size_t size_MB=1024*1024;
      size_t mem_1time=0, mem_2time=0;

      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // Ut
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // hmf
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // Ut
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // hmf
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // Ut
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // hmf
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // Ut
      mem_1time += cntr::mem_function<double>(nt,1)*latt_.lattice_model_.nk_; // hmf

      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
      mem_2time += cntr::mem_herm_matrix<double>(nt,ntau,1)*latt_.lattice_model_.nk_; // G
//      mem_2time += cntr::mem_herm_matrix<double>(Nt,Ntau,Norb)*Nk; // Sigma
//      mem_2time += cntr::mem_herm_matrix<double>(Nt,Ntau,Norb)*Nk; // P
//      mem_2time += cntr::mem_herm_matrix<double>(Nt,Ntau,Norb)*Nk; // W
      // convert to MB
      mem_1time = ceil(mem_1time/(double)size_MB);
      mem_2time = ceil(mem_2time/(double)size_MB);
      //
           std::cout << "Total" << std::endl;
           std::cout << "Hamiltonian : " << mem_1time << " MB" << std::endl;
           std::cout << "Propagators : " << mem_2time << " MB" << std::endl;
      //
      std::cout << "Per rank" << std::endl;
      std::cout << "Hamiltonian : " << mem_1time/double(ntasks) << " MB" << std::endl;
      std::cout << "Propagators : " << mem_2time/double(ntasks) << " MB" << std::endl;
      //
      print_line_minus(50);
      cout << "\n\n";
     
      }
   
//     exit(0);

//    std::cout << "HI" << std::endl;

     cntr::function<double> U_(nt,1),e_loc(nt,1);

            if(tid==tid_root){
                        imp.solve_atomic();
                        latt_.Delta_up.set_timestep_zero(-1);
                        latt_.Delta_up_cc.set_timestep_zero(-1);
                        latt_.Gloc_up.set_timestep_zero(-1);

           for(int tstp=-1;tstp<=nt;tstp++){
            cdmatrix tmp1(1,1),tmp2(1,1);
            tmp1(0,0)=U_val[tstp+1];
            U_.set_value(tstp,tmp1);
            tmp2(0,0)=cdouble(imp.hamiltonian.eps_up[tstp+1],0);     
            e_loc.set_value(tstp,tmp2);
	    //std::cout << tmp2 << std::endl;
         }
          
     }
       for(int tstp=-1;tstp<=nt;tstp++){
       U_.Bcast_timestep(tstp,tid_root);
       e_loc.Bcast_timestep(tstp,tid_root);     
       }

       for(int tstp=-1;tstp<=nt;tstp++){
        latt_.get_h_tilde(tstp,e_loc);
       }

       //if(tid==tid_root){
       //e_loc.print_to_file("Eloc.out");
       //}

//       if(bath_flag){
 
       if(tid==tid_root){

       if(bath_flag){

       bethedos_2 dos;
       
       green_equilibrium_ohmic_bath(latt_.D,dos,bath_beta,h,100,20);

//       latt_.D.print_to_file("D.out");

        for(int tstp=-1;tstp<=nt;tstp++){
         cdmatrix tmp1(1,1);
         tmp1(0,0)=Lamda_[tstp+1];
         latt_.gfunc.set_value(tstp,tmp1);
          
         }
 
        for(int tstp=-1; tstp <= nt; tstp++) {
         latt_.D.left_multiply(tstp, latt_.gfunc);
         latt_.D.right_multiply(tstp, latt_.gfunc);
        }

     
       }

     }
//      if(tid==tid_root) latt_.D.print_to_file("D.out");
  
   //  if(tid==tid_root) latt_.D.print_to_file("D.out");

//      if(tid==tid_root) std::cout << "done" << std::endl; 

//      if(tid==tid_root) e_loc.print_to_file("avarage_out");

//      exit(0);

          if(mix_sigma==1){
             mix_b=true;
           }
//          if(tid==tid_root) std::cout << mix_b << std::endl;

            ppsc::gf_type Gtemp(-1, ntau, 1, -1);
      
            for(iter=1;iter<=MatsMaxIter;iter++){                   
                   if(tid==tid_root) {   
                   ppsc::pp_ints_type pp_ints = get_pp_ints(latt_.Delta_up,latt_.Delta_up_cc,imp.hamiltonian);
                   ppsc::gf_verts_type gf_verts = get_gf_verts(imp.hamiltonian);
                   imp.update_diagrams(pp_ints, gf_verts);
                   imp.pp_step(-1);
                   ppsc::gf_tstps_type gf_tstps = imp.get_spgf(-1);
                   ppsc::gf_tstp_type gloc_old(-1, ntau, 1,-1);
                   ppsc::gf_tstp_type gloc_mix(-1, ntau, 1,-1);
                   ppsc::gf_tstp_type tmp(-1, ntau, 1,-1);
                   gloc_mix.clear();
                   latt_.Gloc_up.get_timestep(-1, gloc_old);
                   tmp.set_matrixelement(0, 0, gf_tstps[0], 0, 0);
                   gloc_mix.incr(tmp, 1.0-Mat_mix);
                   gloc_mix.incr(gloc_old, Mat_mix);
                   latt_.Gloc_up.set_timestep(-1, gloc_mix);
                   cntr::force_matsubara_hermitian(latt_.Gloc_up);
                   latt_.get_glatt(-1,latt_.Gloc_up);
                   latt_.Chi_c_.set_timestep(-1, gf_tstps[1]);
                   latt_.Chi_z_.set_timestep(-1, gf_tstps[2]);
                   cntr::force_matsubara_hermitian(latt_.glatt_);
                   }
                  
                  latt_.Bcast_g0_local(-1,tid_root); 
                  latt_.get_G_K_DMFT(-1);
                  latt_.symmetrize_G_DMFT();
                  latt_.step(-1);
//                  if(tid==tid_root) cntr::force_matsubara_hermitian(latt_.Delta_up);
//                  std::cout << "HI2" << endl;
                   dmfterr = 0.0;
                   if(tid==tid_root) {

                   // cntr::force_matsubara_hermitian(latt_.Delta_up); 

                    if(bath_flag) {
                      cntr::herm_matrix_timestep<double> tmp1(-1, ntau, 1, -1);
                      cntr::Bubble2(-1, tmp1, latt_.Gloc_up, latt_.D);
                      latt_.Delta_up.incr_timestep(-1, tmp1); 
                     }     
             
                    cntr::force_matsubara_hermitian(latt_.Delta_up);                      
 
                    ppsc::set_bwd_from_fwd(-1,latt_.Delta_up_cc,latt_.Delta_up);
                    
                    dmfterr=cntr::distance_norm2(-1,latt_.Gloc_up,Gtemp);
                    Gtemp.set_timestep(-1, latt_.Gloc_up);
                    cout << "DMFT_iter:  " << iter << " err: " << dmfterr  << endl;
                  }
                  MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&dmfterr,1,MPI::DOUBLE,MPI::SUM);
                  if(dmfterr < errmax){
                     matsubara_converged=true;
                                break;
                    }
                } 

               if(iter>MatsMaxIter){
               std::cerr << "WARNING: DMFT Matsubara not converged  after " << MatsMaxIter << "steps ... abort" << std::endl;
//               cerr << "skip real-time calculation " << endl;
               }   


               if(tid==tid_root){
                 latt_.g_loc_.set_timestep(-1,latt_.Gloc_up);
                 latt_.Delta_.set_timestep(-1,latt_.Delta_up);
                }

               latt_.Bcast_g_local(-1,tid_root);
        
           if(tid==tid_root){
             cntr::force_matsubara_hermitian(latt_.Chi_c_);
             cntr::force_matsubara_hermitian(latt_.Chi_z_);
             latt_.get_imp_polarization(-1,U_,latt_.Chi_c_,latt_.Chi_z_);
             cntr::force_matsubara_hermitian(latt_.Pi_c_);
             cntr::force_matsubara_hermitian(latt_.Pi_z_);
            }
   
           latt_.Bcast_Pi_local(-1,tid_root);

           latt_.Get_bare_propagators(-1);

           latt_.symmetrize_Bare_G();
        
           latt_.Get_bare_w_c_q_0(-1);
 
           latt_.Initialize_on_matsbara(-1);
       
         
//           latt_.get_bare_G_les(-1);          

           latt_.Bcast_ret_q_0(-1,tid_root);  
  
           bool matsubara_converged=false;


          for(int iter=1;iter<=MatsMaxIter;iter++){
                   
           err_e = 0.0;
           err_b_c = 0.0;
           err_b_z = 0.0;


           latt_.step_dual(-1,iter,kt,mix_b,err_e,err_b_c,err_b_z);

//           MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_e,1,MPI::DOUBLE,MPI::SUM);
//           MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_b_c,1,MPI::DOUBLE,MPI::SUM);
//           MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_b_z,1,MPI::DOUBLE,MPI::SUM);


              latt_.symmetrize_G_dual();

              if(tid==tid_root){
               std::cout << "tstp= " << -1 << " Dual_space_iter:  " << iter << " err_G_: " << err_e << " err_W_c: " << err_b_c << " err_W_z: " << err_b_z << std::endl;
              }

	     // std::cout << "tstp= " << -1 << " Dual_space_iter:  " << iter << " err_G_: " << err_e << " err_W_c: " << err_b_c << " err_W_z: " << err_b_z << std::endl;
              
              if(err_e < errmax && err_b_c < errmax && err_b_z < errmax && iter>=2) {
                   matsubara_converged=true;
                   break;
                  }
            }
               if(iter>MatsMaxIter){
                        std::cout << "DGW Matsubara didn't converged" << std::endl;
                        abort();
                }

         
         if(nt>=0 && matsubara_converged==true){

            matsubara_converged = false;

            ppsc::gf_type Gtemp1(kt, ntau, 1, -1);   

            for (int n = 0; n <= kt; n++) {
             
               if(tid==tid_root){
                 if(n == 0){
                imp.init_real_time();
                cntr::set_t0_from_mat(latt_.Delta_up);
                cntr::set_t0_from_mat(latt_.Delta_up_cc);
               }
               else{ 
                imp.extrapolate_timestep(n - 1);
                cntr::extrapolate_timestep(n-1,latt_.Delta_up,(kt<n-1 ? kt : n-1));
                cntr::extrapolate_timestep(n-1,latt_.Delta_up_cc,(kt<n-1 ? kt : n-1));
              }
             
             }
            
           }


            for (int iter_warmup = 1; iter_warmup <= itermax; iter_warmup++) {

                if(tid==tid_root){
                   ppsc::pp_ints_type pp_ints = get_pp_ints(latt_.Delta_up,latt_.Delta_up_cc,imp.hamiltonian);
                   ppsc::gf_verts_type gf_verts = get_gf_verts(imp.hamiltonian);
                   imp.update_diagrams(pp_ints, gf_verts);
                   imp.pp_step(kt);
                   for (int n = 0; n <= kt; n++) {
                     ppsc::gf_tstps_type gf_tstps = imp.get_spgf(n);
                    ppsc::gf_tstp_type gloc_old(n, ntau, 1, -1);
                     ppsc::gf_tstp_type gloc_mix(n, ntau, 1, -1);
                     ppsc::gf_tstp_type tmp(n, ntau, 1,-1);
                     gloc_mix.clear();
                     latt_.Gloc_up.get_timestep(n, gloc_old);
                     tmp.set_matrixelement(0, 0, gf_tstps[0], 0, 0);
                     gloc_mix.incr(tmp, 1.0-real_mix);
                     gloc_mix.incr(gloc_old, real_mix);
                     latt_.Gloc_up.set_timestep(n, gloc_mix);
//                     latt_.Gloc_up.set_timestep(n, gf_tstps[0]);
                     latt_.Chi_c_.set_timestep(n, gf_tstps[1]);
                     latt_.Chi_z_.set_timestep(n, gf_tstps[2]);
                   }
            
                for (int n = 0; n <= kt; n++) {
                     latt_.get_glatt(n,latt_.Gloc_up);                 
                   }

                 }

                for (int n = 0; n <= kt; n++) {
                  latt_.Bcast_g0_local(n,tid_root);
                }

                for (int n = 0; n <= kt; n++) {
                  latt_.get_G_K_DMFT(n);
                  latt_.step(n);
               }
          
               dmfterr = 0.0;
               if(tid==tid_root) {

                 for (int n = 0; n <= kt; n++) {

                   if(bath_flag) {
                      cntr::herm_matrix_timestep<double> tmp1(n, ntau, 1, -1);
                      cntr::Bubble2(n, tmp1, latt_.Gloc_up, latt_.D);
                      latt_.Delta_up.incr_timestep(n, tmp1);
                     }

                   ppsc::set_bwd_from_fwd(n,latt_.Delta_up_cc,latt_.Delta_up);

                   dmfterr += cntr::distance_norm2(n, Gtemp1, latt_.Gloc_up);
                   Gtemp1.set_timestep(n, latt_.Gloc_up);
                 }
                cout << "DMFT_WARMUP: iter:  " << iter_warmup << " err: " << dmfterr << endl;
              } 
             MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&dmfterr,1,MPI::DOUBLE,MPI::SUM);

          if (dmfterr < errmax) {
            matsubara_converged = true;
            break;
          }

        } 

        if(tid==tid_root){
          for (int n = 0; n <= kt; n++){ 
            latt_.g_loc_.set_timestep(n,latt_.Gloc_up);
            latt_.Delta_.set_timestep(n,latt_.Delta_up);
          }
        }
       for (int n = 0; n <= kt; n++){ 
            latt_.Bcast_g_local(n,tid_root); 
        }   
 


      if(tid==tid_root){
          for (int n = 0; n <= kt; n++){
            latt_.get_imp_polarization(n,U_,latt_.Chi_c_,latt_.Chi_z_);
          } 
      }


        for (int n = 0; n <= kt; n++){ 
         latt_.Bcast_Pi_local(n,tid_root);
         }

      for (int n = 0; n <= kt; n++){
         latt_.Get_bare_propagators(n);
         latt_.Get_bare_w_c_q_0(n);
         
       }

       for (int n = 0; n <= kt; n++){
         latt_.Bcast_ret_q_0(n,tid_root);
//         latt_.Initialize_on_matsbara(n);
//         latt_.get_bare_G_les(n);
       }


       matsubara_converged=false;

       for (int n = 0; n <= kt; n++) {


        latt_.extrapolate_timestep_dual(n-1,(kt<n-1 ? kt : n-1));
   

        for(int iter_warmup=1; iter_warmup<=itermax; iter_warmup++){
      
           err_e = 0.0;
           err_b_c = 0.0;
           err_b_z = 0.0;

           latt_.step_dual(n,iter_warmup,(n>=kt ? kt : n),mix_b,err_e,err_b_c,err_b_z);
  
//	   MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_e,1,MPI::DOUBLE,MPI::SUM);
//           MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_b_c,1,MPI::DOUBLE,MPI::SUM);
//	   MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&err_b_z,1,MPI::DOUBLE,MPI::SUM);

//           if(tid==tid_root) std::cout << (n>=kt ? kt : n) << std::endl;             
        
           if(tid==tid_root){
               cout << "tstp= " << n << " Dual_space_Warm_up:  " << iter_warmup << " err_G_: " << err_e << " err_W_c: " << err_b_c << " err_W_z: " << err_b_z << endl;
              }

              if(err_e < errmax && err_b_c < errmax && err_b_z < errmax && iter_warmup>=2) {
                   matsubara_converged=true;
                   break;
                  }
            }
               if(iter>MatsMaxIter){
                        cout << "DGW Matsubara didn't converged" << endl;
                       abort();
            }

         }


        
//      if(matsubara_converged == true){

        for (int tstp = kt + 1; tstp <= nt; tstp++) {

           ppsc::gf_tstp_type Gtemp(tstp, ntau, 1, -1);
           
           if(tid==tid_root) {
               imp.extrapolate_timestep(tstp - 1);
               cntr::extrapolate_timestep(tstp-1,latt_.Delta_up,kt);
               cntr::extrapolate_timestep(tstp-1,latt_.Delta_up_cc,kt);
            }

           for (int iter_rt = 1; iter_rt <= iter_rtime; iter_rt++) {

             if(tid==tid_root){   
               ppsc::pp_ints_type pp_ints = get_pp_ints(latt_.Delta_up,latt_.Delta_up_cc,imp.hamiltonian);
               ppsc::gf_verts_type gf_verts = get_gf_verts(imp.hamiltonian);
               imp.update_diagrams(pp_ints, gf_verts);
               imp.pp_step(tstp);
               ppsc::gf_tstps_type gf_tstps = imp.get_spgf(tstp);
               latt_.Gloc_up.set_timestep(tstp, gf_tstps[0]);
               latt_.get_glatt(tstp,latt_.Gloc_up);
               latt_.Chi_c_.set_timestep(tstp, gf_tstps[1]);
               latt_.Chi_z_.set_timestep(tstp, gf_tstps[2]);
              }

             latt_.Bcast_g0_local(tstp,tid_root);
             latt_.get_G_K_DMFT(tstp);
             latt_.step(tstp);
             dmfterr = 0.0;
             if(tid==tid_root) {

               if(bath_flag) {
                      cntr::herm_matrix_timestep<double> tmp1(tstp, ntau, 1, -1);
                      cntr::Bubble2(tstp, tmp1, latt_.Gloc_up, latt_.D);
                      latt_.Delta_up.incr_timestep(tstp, tmp1);
                     }

               ppsc::set_bwd_from_fwd(tstp,latt_.Delta_up_cc,latt_.Delta_up);
               dmfterr =  cntr::distance_norm2(tstp,Gtemp,latt_.Gloc_up);
               Gtemp.set_timestep(tstp, latt_.Gloc_up);
               std::cout << "tstp= " << tstp << " DMFT_Iteration " << iter_rt << ", error: " << dmfterr << std::endl;
              }
             MPI::COMM_WORLD.Allreduce(MPI::IN_PLACE,&dmfterr,1,MPI::DOUBLE,MPI::SUM);
             if(dmfterr < errmax && iter_rt>=2) break;
                
            }
 

          if(tid==tid_root){
            latt_.g_loc_.set_timestep(tstp,latt_.Gloc_up);
            latt_.Delta_.set_timestep(tstp,latt_.Delta_up);
           }

            latt_.Bcast_g_local(tstp,tid_root);

          if(tid==tid_root){
             latt_.get_imp_polarization(tstp,U_,latt_.Chi_c_,latt_.Chi_z_);
          } 

          latt_.Bcast_Pi_local(tstp,tid_root); 

          latt_.Get_bare_propagators(tstp); 
       
          latt_.Get_bare_w_c_q_0(tstp);

          latt_.Bcast_ret_q_0(tstp,tid_root);
        
          latt_.extrapolate_timestep_dual(tstp-1,kt);    

//          cntr::extrapolate_timestep(tstp-1,latt_.f3,kt);

          for(int iter_rt=1;iter_rt<=iter_rtime;iter_rt++){

           err_e = 0.0;
           err_b_c = 0.0;
           err_b_z = 0.0;

           latt_.step_dual(tstp,iter_rt,kt,mix_b,err_e,err_b_c,err_b_z);

          if(tid==tid_root){
               cout << "tstp= " << tstp << " Dual_space_iteration:  " << iter_rt << " err_G_: " << err_e << " err_W_c: " << err_b_c << " err_W_z: " << err_b_z << endl;
              }

          if(err_e < errmax && err_b_c < errmax && err_b_z < errmax && iter_rt>=2) break;
            }
          }
//     }
/**/

 } //real-time lop....

//Calculate lattice real Green's function and susceptibilty: 
          
//           latt_.DMFT_band_structure(); 
          
           latt_.print_DMFT_file();
/**/ 

           {

           if(tid==tid_root) {
            latt_.Gloc_up.print_to_file("G_DMFT_loc.out");
            latt_.Delta_up.print_to_file("D_DMFT_loc.out");
            latt_.Chi_c_.print_to_file("Chi_c_DMFT_loc.out");
            latt_.Chi_z_.print_to_file("Chi_z_DMFT_loc.out");
//	    latt_.glatt_.print_to_file("G_0_imp.out");
            //latt_.Pi_c_.print_to_file("Imp_pola_c.out");
            //latt_.Pi_z_.print_to_file("Imp_pola_z.out");   
//            latt_.retarded_w_charge_.print_to_file("retarded_green_node_0.out");            
           }

//            if(tid==tid_root){
//       std::vector<std::vector<double>> Ekin_oup = get_orb_kinetic_energy(latt_.Gloc_up, latt_.Delta_up, beta, h, kt);
//         ofstream out("Kinetic.out");
//            out.precision(10);
//            for(auto tstp = 0; tstp <= nt+1 ; ++tstp) out << (tstp-1) << "\t" << imp.hamiltonian.docc_exp[tstp]
//              <<  "\t" << imp.hamiltonian.nu_exp[tstp] <<  "\t" << Ekin_oup[0][tstp] << "\t" << imp.hamiltonian.Eint_exp[tstp] << endl;
//       }
   

        //           latt_.DMFT_band_structure();
 
          
           double na,docca,curr,ekin,da;
           FILE *out;
           if(tid==tid_root) out=fopen("obs_DMFT.out","w");
           for(int tstp1=-1;tstp1<=nt;tstp1++){
            double na=latt_.get_dens(tstp1);
            double da=latt_.get_double_DMFT(tstp1); 
            double ekin=latt_.get_ekin(tstp1);
            double curr=latt_.get_current(tstp1);
              if(tid==tid_root) {
                fprintf(out,"t: %d ",tstp1);
                fprintf(out," dens: %.11g",na);
                fprintf(out," docc: %.11g",da);
//                fprintf(out," doccA: %.11g", imp.hamiltonian.docc_exp[tstp1+1]);
//                fprintf(out," E_loc: %.10g", imp.hamiltonian.Eint_exp[tstp1+1]);
                fprintf(out," ekin: %.11g",2.0*ekin);
                fprintf(out," curr: %.11g",curr);
                fprintf(out,"\n");
               }
            }

           if(tid==tid_root) fclose(out); 

           if(tid==tid_root) out=fopen("optical_DMFT.out","w");

            cntr::herm_matrix<double> cond_(nt, ntau, 1, 1);

            for(int tstp1=0;tstp1<=nt;tstp1++){
         
             std::vector<double> sigma(tstp1+1);
             double sdia;

             latt_.optical_conductivity_bubble_DMFT(tstp1, sigma, sdia);              

             for(int n=0;n<=tstp1;n++){
             if(tid==tid_root){
	      cond_.set_ret(tstp1,n,cdouble(sigma[n],0));
              fprintf(out,"t: %i ",tstp1);
              fprintf(out,"t': %i ",n);
	      fprintf(out,"sdia': %.10g ",sdia);
              fprintf(out,"sigma': %.10g ",sigma[n]);
            //  fprintf(out,"sdia': %.10g ",sdia);
              fprintf(out,"\n");
             }

          }

         if(tid==tid_root) fprintf(out,"\n");

       }

       if(tid==tid_root) cond_.print_to_file("conductivity.out");

        if(tid==tid_root) fclose(out);

         }

//	  if(tid==tid_root) std::cout << "D_TRILEX_started" << std::endl; 
          latt_.get_dgw();
//          if(tid==tid_root) std::cout << "D_TRILEX_done" << std::endl;

//          for (int tstp = -1; tstp <= nt; tstp++){
//          latt_.get_local_dual_G(tstp);
//          }

//          if(tid==tid_root) {
//          latt_.G_tilde_local.print_to_file("g_tilde.out");
//          latt_.G_dual_local.print_to_file("g_dual.out");
//          latt_.Pi_c_.print_to_file("Imp_pola_c.out");
//          latt_.Pi_z_.print_to_file("Imp_pola_z.out");
//          }


//           latt_.print_DGW();

	  // for (int tstp = -1; tstp <= nt; tstp++){
	  // latt_.get_local_dual_G(tstp);
          // }

           latt_.clear_data(); 
           
	   latt_.init_for_observables();         

//	   if(tid==tid_root) std::cout << "D_TRILEX_started" << std::endl;


           for (int tstp = -1; tstp <= nt; tstp++){

//            if(tstp==-1){
//            latt_.get_real_G_Chi(tstp,kt,U_);
//            }
//            else{
//            latt_.get_real_G_Chi(tstp,(tstp>=kt ? kt : tstp),U_);
//            latt_.get_real_G_Chi(tstp,(tstp>=kt ? kt : tstp),U_);
//            } 
             latt_.get_real_G_r(tstp,kt,U_);
            // latt_.get_real_G_chi(tstp,kt,U_);
            // latt_.get_sigma_bar(tstp,kt);
            //latt_.get_local_bare(tstp);
           }
  
       
 //          if(tid==tid_root) {
//            latt_.chi_bare_c.print_to_file("screened_c.out");
//            latt_.chi_bare_z.print_to_file("screened_z.out");
//           }
            
          if(tid==tid_root){

           latt_.get_T_matrix_energy(latt_.Sigma_T,latt_.Sigma_H);

           latt_.Sigma_T.print_to_file("DMFT_SIGMA.out");

//           latt_.sigma_corr.print_to_file("DGW_SIGMA.out");      
           }


           double na,docca,curr,ekin;
           FILE *out;
           if(tid==tid_root) out=fopen("obs_Dgw.out","w");
           for(int tstp1=-1;tstp1<=nt;tstp1++){
            double docca=latt_.get_double_occupancy(tstp1);
            double na=latt_.get_Dgw_dens(tstp1);
            double ekin=latt_.get_Dgw_ekin(tstp1);
            double poten=latt_.get_potential_energy(tstp1);  
            double poten_s=latt_.get_potential_singular(tstp1);
            double curr=latt_.get_Dgw_current(tstp1);
            double pot_c=latt_.get_PE_chi_c(tstp1);
              if(tid==tid_root) {
                fprintf(out,"t: %d ",tstp1);
                fprintf(out," dens: %.11g",na);
                fprintf(out," docc: %.10g",docca);
                //current, docc, field, ekin 
                fprintf(out," ekin: %.11g",2.0*ekin);
		fprintf(out," ePEc: %.10g",0.5*pot_c);
                fprintf(out," ePE: %.10g",poten);
                fprintf(out," PEs: %.10g",poten_s);
                fprintf(out," cur: %.10g",curr);
                //fprintf(out," epot: %.10g",pot_c);
		//fprintf(out," curr: %.10g",pot_c.imag());
                fprintf(out,"\n");
                }
             }
             if(tid==tid_root) fclose(out);

	     
	if(tid==tid_root){

        latt_.chi_lc.print_to_file("chi_c_loc.out");

        latt_.chi_lz.print_to_file("chi_z_loc.out");

        latt_.G_l.print_to_file("G_r_loc.out");

        latt_.sigma_corr.print_to_file("DGW_SIGMA.out");

       }

    
       if(tid==tid_root) out=fopen("obs_energy.out","w"); 
       for(int tstp1=-1;tstp1<=nt;tstp1++){
             if(tid==tid_root){
             cdouble xA,xB;
             cntr::convolution_density_matrix(tstp1,&xA,latt_.Sigma_T,latt_.g_loc_,integration::I<double>(kt),beta,h);
             cntr::convolution_density_matrix(tstp1,&xB,latt_.Sigma_T,latt_.G_l,integration::I<double>(kt),beta,h);
             //cdmatrix A1(1,1),A2(1,1),A3(1,1);
             //latt_.Sigma_H.get_value(tstp1,A1);
             //latt_.g_loc_.density_matrix(tstp1,A2);
             //latt_.G_l.density_matrix(tstp1,A3);
             fprintf(out,"t: %i ",tstp1);
             fprintf(out," DMFT_energy: %.10g",xA.real());
             fprintf(out," DGW_energy: %.10g",xB.real());
             //fprintf(out," DMFT_s: %.10g",(A1(0,0).real()*A2(0,0).real()));
             //fprintf(out," DGW_s: %.10g",(A1(0,0).real()*A3(0,0).real()));
             fprintf(out,"\n");
             }

            }

       if(tid==tid_root) fclose(out);
             
             latt_.print_file();

//             latt_.print_sigma_file();

             latt_.Get_matsubara();

            if(tid==tid_root) out=fopen("optical_DGW.out","w");

            for(int tstp1=0;tstp1<=nt;tstp1++){
         
             std::vector<double> sigma(tstp1+1);
             double sdia;

             latt_.optical_conductivity_bubble_DGW(tstp1, sigma, sdia);              

             for(int n=0;n<=tstp1;n++){
             if(tid==tid_root){
              fprintf(out,"t: %i ",tstp1);
              fprintf(out,"t': %i ",n);
	      fprintf(out,"sigma': %.10g ",sdia);
              fprintf(out,"sigma': %.10g ",sigma[n]);
        //      fprintf(out,"D_w': %.10g ",sdia);
              fprintf(out,"\n");
             }

          }

           if(tid==tid_root) fprintf(out,"\n");

       }

        if(tid==tid_root) fclose(out);
/*


      if(tid==tid_root){
       std::vector<std::vector<double>> Ekin_oup = get_orb_kinetic_energy(latt_.Gloc_up, latt_.Delta_up, beta, h, kt);
         ofstream out("Kinetic.out");
            out.precision(10);
            for(auto tstp = 0; tstp <= nt+1 ; ++tstp) out << (tstp-1) << "\t" << imp.hamiltonian.docc_exp[tstp]
              <<  "\t" << imp.hamiltonian.nu_exp[tstp] <<  "\t" << Ekin_oup[0][tstp] << "\t" << imp.hamiltonian.Eint_exp[tstp] << endl; 
       }
*/

  } // try
  catch (char *message) {
    cerr << "exception\n**** " << message << " ****" << endl;
    cerr << "CDMFT input_file [ --test ]\n" << endl;
  } catch (std::exception &e) {
    cerr << "exception: " << e.what() << endl;
    cerr << "\nCDMFT input_file [ --test ]\n" << endl;
  }

  MPI_Finalize();
  return 0;
}




