#pragma once

#include <sys/stat.h>
#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>
#include <sys/types.h>
#include <unistd.h>
#include <cntr/cntr.hpp>
#include <cntr/integration.hpp>
#include "Dgw_lattice.hpp"

namespace Dgw {

//void distribute_kk(int mpi_num,int mpi_imp,int nk1,std::vector<int> &mpi_pid_kk);
//void get_glatt(int nt,int tstp,cntr::herm_matrix<double> &g_latt,cntr::herm_matrix<double> &G1,cntr::herm_matrix<double> &G1c,
//cntr::herm_matrix<double> &Gloc, double beta,double h, int kt1);
//void get_glatt(int nt,int tstp,cntr::herm_matrix<double> &g_latt,cntr::herm_matrix<double> &Delta, cntr::function<double> &e_loc,
//cntr::herm_matrix<double> &G1,cntr::herm_matrix<double> &G1c,cntr::herm_matrix<double> &Gloc,double beta,double h,int kt1);
////////////////////////////////////////////////////////////////////////////////////////
// ERROR HANDLING/DEBUGGING using the cntr_exception
#define CDMFT_ASSERT_0 1  
#define CDMFT_ASSERT_1 0  // for debugging only
#define CPLX std::complex<double>
#define NSWEEP 6
// parallel setup:
#define CDMFT_CAN_USE_MPI 1
#define CDMFT_CAN_USE_OMP 1
template <class LATT> class selfconsistency_pm{
public:
        typedef cntr::herm_matrix_timestep<double> greentstp1x1;
        typedef cntr::herm_matrix<double> green1x1;
        typedef cntr::function<double> cfunc1x1;
        typedef cntr::distributed_timestep_array<double> DIST_TIMESTEP;
        typedef cntr::distributed_array<double> DIST_ARRAY;
        int nc_;   //=NC
        int nt_;
        int ntau_;
        double beta_;
        double h_;
        double nloc_tau_;
        int kt_;
        int nk1_;


        int mpi_num_;   // number of MPI ranks
        int mpi_imp_;   // MPI ranks for the impurity model
        int mpi_pid_;   // local mpi rank (=0) if MPI is not defined
        int omp_num_;

        LATT lattice_model_;

        std::vector< kpoint<LATT> > kk_functions_;        
     
        
        int nklocal_;
        std::vector<dvector> kpoints_local_;
        std::vector<double>  kweight_local_;
        std::vector<int> kindex_rank_;  // kpoints_local[q]=kpoints_[kindex_local[q]]
        std::vector<int>  mpi_pid_kk_;
       
        green1x1 glatt_, g_loc_,retarded_w_charge_;  // on every MPI rank

        green1x1 Gloc_up, Delta_,chi_bare_c,chi_bare_z;
        green1x1 G1_, G1c_;
        green1x1 G2_, D;
        green1x1 G_Delta, Delta_G;
        green1x1 Delta_up, Delta_up_cc;
        green1x1 Pi_c_, Pi_z_;
        green1x1 Chi_c_, Chi_z_, gtmp2_;
        green1x1 G_Sigma, G_Sigma_cc;
        green1x1 TK_bar, G_Sigma_sg_G, Pi_bar_c, Pi_bar_z, w_bare_c, w_bare_z;
        green1x1 g_temp,g_temp_cc,Sigma_T,sigma_corr, G_tilde_local, G_dual_local;
        green1x1 g_bose1, g_bose1_cc, g_bose2, g_bose2_cc;
        green1x1 chi_lc, chi_lz, G_l, G_DMFT, loc_pi_c, loc_pi_lat_c, loc_pi_z, loc_pi_lat_z;        
 
        cfunc1x1 ret_U_,tv_G_s_,f3,dress_f,Sigma_H,gfunc;

        DIST_TIMESTEP gk_all_timesteps_, wk_c_all_timesteps_, wk_z_all_timesteps_;
        DIST_ARRAY convergence_error_e_, convergence_error_c_,  convergence_error_z_;
        DIST_ARRAY  G_r_r_, W_c_r_,W_z_r_;
        DIST_ARRAY  G_r_i_, W_c_i_,W_z_i_;
        DIST_ARRAY  G_DMFT_r_, G_DMFT_i_;
//        cfunc1x1 U_;

        void init_parallel_SINGLE(void);
        void init_parallel_MPI(int mpi_imp); // use MPI to distribute kpoints

        void set_OMP(int omp_num=-1);

        void print_mem_layout(FILE *out,int nt,int ntau);
        // this assumes that the lattice_model and parallel setup is initialized:
        void init(int nt,int ntau,double beta,double h,int kt);
        void set_kt(int kt);
       
//        void extrapolate_Delta(int tstp);

        void step(int tstp);
 
        void Get_bare_w_c_q_0(int tstp);

        double get_dens( int tstp);

        double get_ekin( int tstp);

        double get_current(int tstp);
                 
        void get_glatt(int tstp, cntr::herm_matrix<double> &Gloc);

        void get_G_K_DMFT(int tstp);

        void Bcast_g(int tstp, int mpi_imp);

        void Bcast_g0_local(int tstp, int mpi_imp);

        void Bcast_g_local(int tstp, int mpi_imp);

        void Bcast_Pi_local(int tstp, int mpi_imp);
 
        void Bcast_ret_q_0(int tstp, int mpi_imp);
  
        void get_imp_polarization(int tstp, cntr::function<double> &U_, 
               cntr::herm_matrix<double> &chi_c, cntr::herm_matrix<double> &chi_z);

//        void Get_Gk_Wk_bare(int tstp);
        void step_dual_Matsubara(int tstp, double &dual_mix , double &err_e, double &err_b_c, double &err_b_z);
     
        void step_dual(int tstp, int &iter, int &kt2, bool &mix, double &err_e, double &err_b_c, double &err_b_z);

        void gather_Gk_dual_timestep(int tstp);

        void gather_Wk_dual_timestep(int tstp);

       // void Get_latt_bubble(int tstp, int qq, cntr::herm_matrix<double> &P, DIST_TIMESTEP &gk_all_timesteps_);
        void Get_latt_bubble(int tstp, int qq, cntr::herm_matrix<double> &P);

       // void get_Sigma_Hartree(int tstp,int kk, cntr::function<double>&S, DIST_TIMESTEP &gk_all_timesteps_); 
        void get_Sigma_Hartree(int tstp,int kk, cntr::function<double>&S);

       // void get_Sigma_dual(int tstp,int kk,cntr::herm_matrix<double> &P, DIST_TIMESTEP &gk_all_timesteps_, 
       //       DIST_TIMESTEP &wk_c_all_timesteps_,DIST_TIMESTEP &wk_z_all_timesteps_);
        void get_Sigma_dual(int tstp,int kk,cntr::herm_matrix<double> &P);

        double get_dual_local_dens(int tstp);

        cdouble get_dual_local_tv(int tstp);

        double get_dual_local_bare_dens(int tstp);

        void Get_G_dual_bare(int tstp,cntr::herm_matrix<double> &S, cntr::herm_matrix<double> &P);
 
        void Get_W_dual_bare(int tstp, cntr::function<double> &f_c, cntr::function<double> &f_z, 
                        cntr::herm_matrix<double> &W_c, cntr::herm_matrix<double> &W_z);

        void get_retarded_U(int tstp, int &iter,cntr::herm_matrix<double> &gtmp2,cntr::function<double> &ret_U_);

        void B_cast_ret_U(int tstp,int mpi_imp_);
    
        void Get_bare_propagators(int tstp);
 
        void Get_initialize_zero(int tstp,  cntr::herm_matrix<double>&S, 
                   cntr::herm_matrix<double> &P, cntr::herm_matrix<double> &Q);
      
        void get_mixed_self_energy(int tstp, int &iter, cntr::herm_matrix<double>&S,
                          cntr::herm_matrix<double> &Q);

        void get_mixed_pi_energy(int tstp, int &iter, cntr::herm_matrix<double>&S,
                          cntr::herm_matrix<double> &Q);
 
        void extrapolate_timestep_dual(int tstp,int solve_order);

        void symmetrize_G_DMFT();
//        void get_real_G(int tstp);

//        void get_real_chi(int tstp,cntr::function<double> &U_);

        double get_double_occupancy(int tstp);

        double get_dual_local_G_les(int tstp);

        void clear_data();
       
        void init_for_observables();

        void get_dgw();

        void print_file();

	void print_DGW();

       double get_Dgw_dens(int tstp);

       double get_average_disper(int tstp); 

       double get_Dgw_ekin( int tstp);

       double get_Dgw_current(int tstp);
     
       void Get_matsubara();
    
       void DMFT_band_structure();
 
       void print_DMFT_file();
  
       void print_sigma_file();

       void symmetrize_G_dual();

       void symmetrize_Bare_G(); 
//       void get_local(int tstp);

       void get_local_bare(int tstp);

       void Initialize_on_matsbara(int tstp);

       void get_bare_G_les(int tstp);
       
       void get_real_G_r(int tstp, int &kt1, cntr::function<double> &U_);

       void get_real_G_chi(int tstp, int &kt1, cntr::function<double> &U_); 

       void get_sigma_bar(int tstp, int &kt1);

       double get_potential_energy(int tstp);

       double get_potential_singular(int tstp);

       double get_double_DMFT(int tstp);

       void get_T_matrix_energy(cntr::herm_matrix<double>&, cntr::function<double> &);
       
       void optical_conductivity_bubble_DMFT(int tstp, std::vector<double> &sigma, double &sdia);

       void optical_conductivity_bubble_DGW(int tstp, std::vector<double> &sigma, double &sdia);

       void get_h_tilde(int tstp, cntr::function<double> &e_loc);

        double get_PE_chi_c(int tstp1);  
        void get_local_dual_G(int tstp);

        void Get_bare_w_propagators(int tstp,cntr::function<double> &,cntr::herm_matrix<double>&,cntr::herm_matrix<double>&);

	void Bcast_bare_w_local(int tstp, int mpi_imp);

	void Initialize_on_boot_strap(int tstp);
//       void print_DMFT_file(){

};


} //namespace 
