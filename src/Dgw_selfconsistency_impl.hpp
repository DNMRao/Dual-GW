#include <sys/stat.h>
#include <complex>
#include <cntr/cntr.hpp>
//#include "Dgw_lattice.hpp"
#include "Dgw_selfconsistency_decl.hpp"
#include "./ppsc/ppsc.hpp"
using namespace cntr;

#define CINTEG integration::I<double>

namespace Dgw {

template <class LATT>
void selfconsistency_pm<LATT>::init_parallel_SINGLE(void){
        mpi_num_=1;
	mpi_pid_=0;
	mpi_imp_=0;
	omp_num_=1;
}
template <class LATT>
void selfconsistency_pm<LATT>::init_parallel_MPI(int mpi_imp){
#ifdef CDMFT_CAN_USE_MPI
	mpi_num_=MPI::COMM_WORLD.Get_size();
	mpi_pid_=MPI::COMM_WORLD.Get_rank(); 
	mpi_imp_=mpi_imp;
	omp_num_=1;
      

#else
	std::cerr << "Calling " << __PRETTY_FUNCTION__ << std::endl;
	std::cerr << "MPI not defined" << std::endl;
#endif
}
template <class LATT>
void selfconsistency_pm<LATT>::set_OMP(int omp_num){
    // ** CURRENTLY OMP NOT YET IMPLEMENTED ** 
#ifdef CDMFT_CAN_USE_OMP
	if(omp_num==-1){
		omp_num_=omp_get_max_threads(); // OMP_NUM_THREADS
	}else{
		omp_num_=omp_num;
	}
#else
	omp_num_=1;
#endif
}
template <class LATT>
void selfconsistency_pm<LATT>::set_kt(int kt){
	kt_=kt;	
}
template <class LATT>
void selfconsistency_pm<LATT>::init(int nt,int ntau,double beta,double h,int kt){
	nt_=nt;
	ntau_=ntau;
	beta_=beta;
	h_=h;
	set_kt(kt);

        nk1_=lattice_model_.nk_;

        gk_all_timesteps_=DIST_TIMESTEP(nk1_,nt_,ntau_,1,-1,true);
        wk_c_all_timesteps_=DIST_TIMESTEP(nk1_,nt_,ntau_,1,+1,true);
        wk_z_all_timesteps_=DIST_TIMESTEP(nk1_,nt_,ntau_,1,+1,true);
        
        mpi_pid_kk_=gk_all_timesteps_.data().tid_map();
        mpi_pid_=gk_all_timesteps_.tid();
        mpi_imp_=0;

	// local functions are stored here:
	if(mpi_pid_==mpi_imp_){
		Gloc_up = green1x1(nt_,ntau_,1,-1);		
		G1_ = green1x1(nt_,ntau_,1,-1);	
		G1c_ = green1x1(nt_,ntau_,1,-1);	
		G2_ = green1x1(nt_,ntau_,1,-1);	
		Delta_up = green1x1(nt_,ntau_,1,-1);
                Delta_up_cc = green1x1(nt_,ntau_,1,-1);
                G_Delta = green1x1(nt_,ntau_,1,-1);
                Delta_G = green1x1(nt_,ntau_,1,-1);
              
                Chi_c_ = green1x1(nt_,ntau_,1,1);
                Chi_z_ = green1x1(nt_,ntau_,1,1);

               Sigma_T = green1x1(nt_,ntau_,1,-1);
               Sigma_H = cfunc1x1(nt_,1);
                 
               D = green1x1(nt_,ntau_,1,1); 

               gfunc = cfunc1x1(nt_,1);

//	       G_tilde_local = green1x1(nt_,ntau_,1,-1);

//	       G_dual_local =  green1x1(nt_,ntau_,1,-1);
//                chi_bare_c = green1x1(nt_,ntau_,1,1);
//                chi_bare_z = green1x1(nt_,ntau_,1,1);             
                 
	}
	// needed on every mpi rank

	glatt_=green1x1(nt_,ntau_,1,-1);
        g_loc_ = green1x1(nt_,ntau_,1,-1);
        Delta_ = green1x1(nt_,ntau_,1,-1);

        Pi_c_ = green1x1(nt_,ntau_,1,1);
        Pi_z_ = green1x1(nt_,ntau_,1,1);
//        gtmp2_ = green1x1(nt_,ntau_,1,1);

       retarded_w_charge_ = green1x1(nt_,ntau_,1,1);  
       ret_U_ = cfunc1x1(nt_,1);     
       tv_G_s_ = cfunc1x1(nt_,1);
       f3 = cfunc1x1(nt_,1);
//       dress_f = cfunc1x1(nt_,1);

        convergence_error_e_ = DIST_ARRAY(nk1_,1,true);
        convergence_error_c_ = DIST_ARRAY(nk1_,1,true);
        convergence_error_z_ = DIST_ARRAY(nk1_,1,true);

       convergence_error_e_.reset_blocksize(1);
       convergence_error_c_.reset_blocksize(1);
       convergence_error_z_.reset_blocksize(1);

//        G_DMFT_r_ = DIST_ARRAY(nk_,1,true);
//        G_DMFT_i_ = DIST_ARRAY(nk_,1,true);
   
        kk_functions_.resize(nk1_);	
	
       for(int q=0;q<nk1_;q++){
           if(mpi_pid_kk_[q]==mpi_pid_){
             kk_functions_[q].init(nt_,ntau_,1,beta_,h_,lattice_model_.kpoints_[q],lattice_model_);
            }	
         }
		

 }
template <class LATT>
void selfconsistency_pm<LATT>::print_mem_layout(FILE *out,int nt,int ntau){
    int nk1=lattice_model_.nk_;
	int q,pid;
	size_t mem_latt,mem_g=cntr::mem_herm_matrix<double>(nt,ntau,1);
	std::vector<int> nkloc(mpi_num_,0);
	for(q=0;q<nk1;q++) nkloc[mpi_pid_kk_[q]]++;
	for(pid=0;pid<mpi_num_;pid++){
		mem_latt=mem_g*(1+nkloc[pid]);
		if(pid==mpi_imp_) mem_latt += mem_g*5;
		fprintf(out,"pid= %d\tkpoints_local= %d\tmem=%g MB\n",pid,nkloc[pid],mem_latt/1024.0/1024.0);
	}
}
template <class LATT>
void selfconsistency_pm<LATT>::get_h_tilde(int tstp, cntr::function<double> &e_loc){

  cdmatrix A1(1,1),A2(1,1),A3(1,1);
   int nk=lattice_model_.nk_;
   for(int q=0;q<nk;q++){
     if(mpi_pid_kk_[q]==mpi_pid_){
         A3.setZero();
//       cdmatrix A1(1,1),A2(1,1),A3(1,1);  
       kk_functions_[q].hk_.get_value(tstp,A1);
       e_loc.get_value(tstp,A2);
       A3=A1-A2;
       kk_functions_[q].hktilde_.set_value(tstp,A3);
//       char filename1[1024];
//       std::sprintf(filename1,"DATA_check/hk_sig_%d.out",q);
//       kk_functions_[q].hktilde_.print_to_file(filename1);
      }
   }
}

template <class LATT>
void selfconsistency_pm<LATT>::get_G_K_DMFT(int tstp){

   int nk=lattice_model_.nk_;
        for(int q=0;q<nk;q++){
          if(mpi_pid_kk_[q]==mpi_pid_) kk_functions_[q].Get_G_latt_DMFT(tstp,kt_,glatt_);
        }

}

template <class LATT>
void selfconsistency_pm<LATT>::symmetrize_G_DMFT(){

   int nk=lattice_model_.nk_;
    for(int q=0;q<nk;q++){  
     if(mpi_pid_kk_[q]==mpi_pid_)cntr::force_matsubara_hermitian(kk_functions_[q].Gk_DMFT);
     }

}
template <class LATT>
void selfconsistency_pm<LATT>::step(int tstp){
 
      int n;

      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

//      int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);

//      int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//      int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);


     for(n=n1;n<=n2;n++){
//      for(n=-1;n<=n2;n++){
      greentstp1x1 temp(n,ntau_,1,-1);
      greentstp1x1 gloc_tstp(n,ntau_,1,-1);
      greentstp1x1 g1_tstp(n,ntau_,1,-1);
      greentstp1x1 g1c_tstp(n,ntau_,1,-1);
      greentstp1x1 g2_tstp(n,ntau_,1,-1);
      gloc_tstp.clear();
      g1_tstp.clear();
      g1c_tstp.clear();
      g2_tstp.clear();
      temp.clear();
       for(int kk=0;kk<lattice_model_.nk_bz_;kk++){
          double wt=lattice_model_.kweight_bz_[kk];
          int q=lattice_model_.idx_kk_[kk];
          if(mpi_pid_kk_[q]==mpi_pid_){
            kk_functions_[q].Gk_DMFT.get_timestep(n,temp);
            gloc_tstp.incr(temp,wt);
            temp.left_multiply(kk_functions_[q].hktilde_);
//            temp.left_multiply(kk_functions_[q].hk_);
            g1_tstp.incr(temp,wt);
            kk_functions_[q].Gk_DMFT.get_timestep(n,temp);
            temp.right_multiply(kk_functions_[q].hktilde_);
//            temp.right_multiply(kk_functions_[q].hk_);
            g1c_tstp.incr(temp,wt);
            temp.left_multiply(kk_functions_[q].hktilde_);
//            temp.left_multiply(kk_functions_[q].hk_);
            g2_tstp.incr(temp,wt);
          }

       } 

        
        gloc_tstp.Reduce_timestep(mpi_imp_);
        g1_tstp.Reduce_timestep(mpi_imp_);
        g1c_tstp.Reduce_timestep(mpi_imp_);
        g2_tstp.Reduce_timestep(mpi_imp_); 
      
       if(mpi_pid_==mpi_imp_){ 
       Gloc_up.set_timestep(n,gloc_tstp);
       G1_.set_timestep(n,g1_tstp);
       G1c_.set_timestep(n,g1c_tstp);
       G2_.set_timestep(n,g2_tstp);
       }

 }

     if(mpi_pid_==mpi_imp_){
  
    
     if(tstp==-1){
                cntr::vie2_mat_fixpoint(Delta_up,G1_,G1c_,G2_,beta_,integration::I<double>(kt_),10);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Delta_up);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Delta_up,G1_,G1c_,G2_,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Delta_up,G1_,G1c_,G2_,beta_,h_,kt_);

         }

	 
//	     cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);
        
//        tmpQsin.set_zero();
//        tmpFsin.set_zero();
//        tmpGsin.set_zero();

//       cntr::vie2_timestep_sin(tstp,Delta_up,tmpGsin,G1_,G1c_,tmpFsin,G2_,tmpQsin,beta_,h_,kt_);

   
      
     
     }  
/**/

 }

//template <class LATT>
//void selfconsistency_pm<LATT>::get_glatt(int tstp, cntr::herm_matrix<double> &Gloc, cntr::function<double> &e_loc){

  
//     int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);
//     Dgw::get_glatt(nt_,kt_,tstp,glatt_,Delta_up,G_Delta,Delta_G,Gloc,beta_,h_,(tstp==-1 || tstp>kt_ ? kt_ : tstp));
//     Dgw::get_glatt(nt_,tstp,glatt_,Delta_up,e_loc,G_Delta,Delta_G,Gloc,beta_,h_,kt_);


//}
template <class LATT>
void selfconsistency_pm<LATT>::Bcast_g0_local(int tstp, int tid_root){
      int n;
      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

      for(n=n1;n<=n2;n++){
         glatt_.Bcast_timestep(n,tid_root);
      }
}

template <class LATT>
void selfconsistency_pm<LATT>::Bcast_ret_q_0(int tstp, int tid_root){

      int n;
      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

      for(n=n1;n<=n2;n++){
      retarded_w_charge_.Bcast_timestep(n,tid_root);
      }

}

template <class LATT>
void selfconsistency_pm<LATT>::Bcast_g_local(int tstp, int tid_root){
      int n;

      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

      for(n=n1;n<=n2;n++){
         g_loc_.Bcast_timestep(n,tid_root);
         Delta_.Bcast_timestep(n,tid_root);
      }
}

template <class LATT>
void selfconsistency_pm<LATT>::Bcast_Pi_local(int tstp, int tid_root){

      int n;
      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

      for(n=n1;n<=n2;n++){
         Pi_c_.Bcast_timestep(n,tid_root);
         Pi_z_.Bcast_timestep(n,tid_root);
      }
}

template <class LATT>
void selfconsistency_pm<LATT>::Get_bare_propagators(int tstp){


for(int q=0;q<nk1_;q++){
     if(mpi_pid_kk_[q]==mpi_pid_){
       Get_G_dual_bare(tstp,kk_functions_[q].Gk_tilde,kk_functions_[q].Gk_DMFT);
       Get_W_dual_bare(tstp,kk_functions_[q].Vertex_c,kk_functions_[q].Vertex_z,
                 kk_functions_[q].Wk_bare_c,kk_functions_[q].Wk_bare_z);
       if(tstp==0) {
	       cntr::set_t0_from_mat(kk_functions_[q].Gk_tilde);
               cntr::set_t0_from_mat(kk_functions_[q].Wk_bare_c);
               cntr::set_t0_from_mat(kk_functions_[q].Wk_bare_z);
	}
      }
     
  }

}

template <class LATT>
void selfconsistency_pm<LATT>::Get_bare_w_c_q_0(int tstp){

// if(mpi_pid_kk_[0]==mpi_pid_){
   if(mpi_pid_==mpi_imp_){
   //if(tstp==-1) std::cout << mpi_pid_kk_[0] << std::endl;
//   std::cout << q << mpi_pid_kk_[0] << std::endl;
   kk_functions_[0].Wk_bare_c.get_timestep(tstp,retarded_w_charge_);
  }
}

template <class LATT>
void selfconsistency_pm<LATT>::symmetrize_Bare_G(){

     //int nk=lattice_model_.nk_;    
     for(int q=0;q<nk1_;q++){
       if(mpi_pid_kk_[q]==mpi_pid_){ 
        cntr::force_matsubara_hermitian(kk_functions_[q].Gk_tilde);
        cntr::force_matsubara_hermitian(kk_functions_[q].Wk_bare_c);
        cntr::force_matsubara_hermitian(kk_functions_[q].Wk_bare_z);
      }

   }
}
template <class LATT>
void selfconsistency_pm<LATT>::Initialize_on_matsbara(int tstp){
for(int q=0;q<nk1_;q++){
  if(mpi_pid_kk_[q]==mpi_pid_){
  kk_functions_[q].Gk_dual.set_timestep(tstp,kk_functions_[q].Gk_tilde);
  kk_functions_[q].Wk_dual_c.set_timestep(tstp,kk_functions_[q].Wk_bare_c);
  kk_functions_[q].Wk_dual_z.set_timestep(tstp,kk_functions_[q].Wk_bare_z);
  } 
 }
}
/*
template <class LATT>
void selfconsistency_pm<LATT>::get_bare_G_les(int tstp){
  cdmatrix At4(1,1);
  At4(0,0)=get_dual_local_G_les(tstp);
  f3.set_value(tstp,At4);
}
*/
template <class LATT>
void selfconsistency_pm<LATT>::extrapolate_timestep_dual(int tstp,int solve_order){

//    int nk=lattice_model_.nk_;
    if(tstp==-1){
         for(int q=0;q<nk1_;q++){
           if(mpi_pid_kk_[q]==mpi_pid_){
            cntr::set_t0_from_mat(kk_functions_[q].Gk_dual);
            cntr::set_t0_from_mat(kk_functions_[q].Wk_dual_c);
            cntr::set_t0_from_mat(kk_functions_[q].Wk_dual_z);

         }
       }
      }else if(tstp>=0){
         for(int q=0;q<nk1_;q++){
          if(mpi_pid_kk_[q]==mpi_pid_){
            cntr::extrapolate_timestep(tstp,kk_functions_[q].Gk_dual,solve_order);
            cntr::extrapolate_timestep(tstp,kk_functions_[q].Wk_dual_c,solve_order);
            cntr::extrapolate_timestep(tstp,kk_functions_[q].Wk_dual_z,solve_order);
         }
      }
   }


}

template <class LATT>
void selfconsistency_pm<LATT>::step_dual(int tstp, int &iter, int &kt2, bool &mix, double &err_e, 
                  double &err_b_c, double &err_b_z){

         double err1, err2, err3;

//	 double err_e_;
//         double err_b_c_;
//         double err_b_z_;

          
         int n;
         int kt1=kt2;

         int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
         int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);
        
//         int n1=(tstp==-1 || tstp>kt1 ? tstp : 0);
//         int n2=(tstp==-1 || tstp>kt1 ? tstp : kt1);    

         for(n=n1;n<=n2;n++){
          gather_Gk_dual_timestep(n);        
          for(int q=0;q<nk1_;q++){
           if(mpi_pid_kk_[q]==mpi_pid_){
             Get_latt_bubble(n,q,kk_functions_[q].Pi_dual);
	     kk_functions_[q].step_Wk_dual_with_error(n,kt_,err1,err2);
	     convergence_error_c_.block(q)[0]=err1;
	     convergence_error_z_.block(q)[0]=err2;
             if((n==-1)&&(mix==1))get_mixed_pi_energy(n,iter,kk_functions_[q].Pi_dual,
                          kk_functions_[q].Pi_mix);
              }
	     
            }

	   convergence_error_c_.mpi_bcast_all();
           convergence_error_z_.mpi_bcast_all();
      
          gather_Wk_dual_timestep(n);
	  if(n1!=n2) gather_Gk_dual_timestep(n);
	  for(int k=0;k<nk1_;k++){
            if(mpi_pid_kk_[k]==mpi_pid_){
           get_Sigma_Hartree(n,k,kk_functions_[k].Hartree_);
           get_Sigma_dual(n,k,kk_functions_[k].Sigma_dual);
           if((n==-1)&&(mix==1)) get_mixed_self_energy(n,iter,kk_functions_[k].Sigma_dual,
                          kk_functions_[k].Sigma_dual_mix);
	    }
          
	  }
 
         cdmatrix A1(1,1);
	 A1.setZero();
         ret_U_.set_value(n,A1);
         get_retarded_U(n,iter,retarded_w_charge_,ret_U_);
         

         cdmatrix A2(1,1), A3(1,1),tmp1(1,1),tmp2(1,1);

         for(int k=0;k<nk1_;k++){
              if(mpi_pid_kk_[k]==mpi_pid_){
                 tmp1.setZero();
                 tmp2.setZero();
                 ret_U_.get_value(n,A2);
                 kk_functions_[k].Hartree_.get_value(n,A3);
                 tmp1(0,0) = A3(0,0) + A2(0,0); // modify
                 tmp2=tmp1.adjoint();
                 tmp1=(tmp1+tmp2)*0.5;
                 kk_functions_[k].Sigma_sg_.set_value(n,tmp1);
                }

	     }

	    for(int k=0;k<nk1_;k++){
               if(mpi_pid_kk_[k]==mpi_pid_){
               kk_functions_[k].step_Gk_dual_with_error(n,kt_,err3);
               convergence_error_e_.block(k)[0]=err3;
              }
            }
            convergence_error_e_.mpi_bcast_all();

           err_e=0.0;
           err_b_c=0.0;
           err_b_z=0.0;


           for(int k=0;k<nk1_;k++) {
            err_e+=convergence_error_e_.block(k)[0];
            err_b_c+=convergence_error_c_.block(k)[0];
            err_b_z+=convergence_error_z_.block(k)[0];
          }


        }

//        err_e = err_e_;
//        err_b_c = err_b_c_;
//	err_b_z = err_b_z_;
           
    //      for(int q=0;q<nk1_;q++){
    //       if(mpi_pid_kk_[q]==mpi_pid_){
    //        kk_functions_[q].step_Wk_dual_with_error(tstp,kt_,err1,err2);
    //         convergence_error_c_.block(q)[0]=err1;
    //         convergence_error_z_.block(q)[0]=err2;

    //         }
    //      }


    //    convergence_error_c_.mpi_bcast_all();
    //    convergence_error_z_.mpi_bcast_all();
 
         
    //      for(n=n1;n<=n2;n++){
    //       gather_Wk_dual_timestep(n);
    //       if(n1!=n2) gather_Gk_dual_timestep(n);
    //       for(int k=0;k<nk1_;k++){
    //        if(mpi_pid_kk_[k]==mpi_pid_){
    //       get_Sigma_Hartree(n,k,kk_functions_[k].Hartree_);
    //       get_Sigma_dual(n,k,kk_functions_[k].Sigma_dual);
           //if((n==-1)&&(mix==1)) get_mixed_self_energy(n,iter,kk_functions_[k].Sigma_dual,
           //               kk_functions_[k].Sigma_dual_mix);
         
    //      }

    //    }

    //  }

       
    //         cdmatrix A1(1,1);
    //         for(n=n1;n<=n2;n++){
//             if(n1!=n2)gather_Gk_dual_timestep(n);   
    //         A1.setZero();
    //         ret_U_.set_value(n,A1);
    //         get_retarded_U(n,iter,retarded_w_charge_,ret_U_);
    //        }
           
    //        cdmatrix A2(1,1), A3(1,1),tmp1(1,1),tmp2(1,1);          
    //        for(n=n1;n<=n2;n++){
    //         for(int k=0;k<nk1_;k++){
    //          if(mpi_pid_kk_[k]==mpi_pid_){
    //             tmp1.setZero();
    //             tmp2.setZero();
    //             ret_U_.get_value(n,A2);
    //             kk_functions_[k].Hartree_.get_value(n,A3);
    //             tmp1(0,0) = A3(0,0) + A2(0,0); // modify
    //		 tmp2=tmp1.adjoint();
    //             tmp1=(tmp1+tmp2)*0.5;
//		 tmp1.setZero();
    //             kk_functions_[k].Sigma_sg_.set_value(n,tmp1);
//                 tmp2=tmp1.adjoint();
//                 kk_functions_[k].Sigma_sd_.set_value(n,tmp2);
      //           }
      //         }
      //      }
/**/
          
      //  for(int k=0;k<nk1_;k++){
      //      if(mpi_pid_kk_[k]==mpi_pid_){
          //   char filename[1024];
          //  std::sprintf(filename,"Singular%d.out",k);
          //   kk_functions_[k].Sigma_sg_.print_to_file(filename);
       //      kk_functions_[k].step_Gk_dual_with_error(tstp,kt_,err3);
       //      convergence_error_e_.block(k)[0]=err3;
       //     }
      //  }
    

      //  convergence_error_e_.mpi_bcast_all();

      //  err_e=0.0;
      //  err_b_c=0.0;
      //  err_b_z=0.0;
   
      // for(int k=0;k<nk1_;k++) {
      //      err_e+=convergence_error_e_.block(k)[0];
      //      err_b_c+=convergence_error_c_.block(k)[0];
      //      err_b_z+=convergence_error_z_.block(k)[0];
     //  }
/**/  
}
template <class LATT>
void selfconsistency_pm<LATT>::get_mixed_self_energy(int tstp, int &iter, cntr::herm_matrix<double> &S, 
                        cntr::herm_matrix<double> &Q){
    
   GREEN_TSTP sigma1(tstp,ntau_,1,-1);
   if(iter==1) Q.set_timestep(tstp,S);
   sigma1.incr(S,0.5);
   sigma1.incr(Q,0.5);
   S.set_timestep(tstp,sigma1); 
   Q.set_timestep(tstp,sigma1);

}

template <class LATT>
void selfconsistency_pm<LATT>::get_mixed_pi_energy(int tstp, int &iter, cntr::herm_matrix<double> &S, 
         cntr::herm_matrix<double> &Q){

   GREEN_TSTP sigma1(tstp,ntau_,1,1);
   if(iter==1) Q.set_timestep(tstp,S);
   sigma1.incr(S,0.5);
   sigma1.incr(Q,0.5);
   S.set_timestep(tstp,sigma1);
   Q.set_timestep(tstp,sigma1);

}

template <class LATT>
void selfconsistency_pm<LATT>::symmetrize_G_dual(){

    int nk=lattice_model_.nk_; 
     for(int q=0;q<nk;q++){
       if(mpi_pid_kk_[q]==mpi_pid_){
        cntr::force_matsubara_hermitian(kk_functions_[q].Pi_dual);
        cntr::force_matsubara_hermitian(kk_functions_[q].Wk_dual_c);
        cntr::force_matsubara_hermitian(kk_functions_[q].Wk_dual_z);
        cntr::force_matsubara_hermitian(kk_functions_[q].Gk_dual);
        cntr::force_matsubara_hermitian(kk_functions_[q].Sigma_dual);

      }

   }

}

template<class LATT>
void selfconsistency_pm<LATT>::B_cast_ret_U(int tstp,int tid_root){

      int n;
//      int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);
      int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

     for(n=n1;n<=n2;n++){
        ret_U_.Bcast_timestep(n,tid_root);
      }
}
template<class LATT>
void selfconsistency_pm<LATT>::get_retarded_U(int tstp, int &iter, cntr::herm_matrix<double> &S, cntr::function<double> &f){
  
  cdmatrix At(1,1),At4(1,1);
  At.setZero();
  At4.setZero();
   int n;
   int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);
//   int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
   int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);
//   int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);
   for(n=-1;n<=n2;n++){
    double loc_G = get_dual_local_G_les(n);
    At4(0,0) = CPLX(loc_G, 0);
    f3.set_value(n,At4);
    }

   
   if(tstp==-1){
     std::complex<double> cc_m,cc_d;
     double dtau_= beta_ / ntau_;
     cdmatrix tmp1(1,1),At1(1,1);
     cntr::response_integrate(ntau_,dtau_,cc_m,S.matptr(0),1,0,integration::I<double>(kt_));
     cc_d = cc_m * f3.ptr(-1)[0];
     double shift=(lattice_model_.U0_[tstp+1]+(8.0*lattice_model_.V_[tstp+1]));
     At1(0,0)=(2.0*cc_d);
     tmp1(0,0)=(shift*f3.ptr(-1)[0]);
     At(0,0) = At1(0,0) + tmp1(0,0);
 }else if(tstp >= kt_){
     std::complex<double> cc_m,cc_d;
     double dtau_= beta_ / ntau_;
     cdmatrix At1(1,1),tmp1(1,1);
     cntr::response_integrate(ntau_,dtau_,cc_m,S.tvptr(tstp,0),1,0,integration::I<double>(kt_));
     cc_d = cc_m * (CPLX(0, -1.0) * f3.ptr(-1)[0]);
     cntr::response_integrate(tstp, h_, cc_m, S.retptr(tstp, 0),1,0, f3.ptr(0),1,0,integration::I<double>(kt_));
     cc_d += cc_m;
     double shift=(lattice_model_.U0_[tstp+1]+(8.0*lattice_model_.V_[tstp+1]));
     At1(0,0) = (2.0*cc_d);
     tmp1(0,0)=(shift*f3.ptr(tstp)[0]);
     At(0,0) = At1(0,0) + tmp1(0,0);
} else{
     std::complex<double> cc_m,cc_d;
     double dtau_= beta_ / ntau_;
     cdmatrix At1(1,1),tmp1(1,1);
     cntr::response_integrate(ntau_,dtau_,cc_m,S.tvptr(tstp,0),1,0,integration::I<double>(kt_));
     cc_d = cc_m * (CPLX(0, -1.0) * f3.ptr(-1)[0]);
     CPLX *wtmp = new CPLX[kt_ + 1];
        for (int i = 0; i <= tstp; i++)
            wtmp[i] = S.retptr(tstp, i)[0];
        for (int i = tstp + 1; i <= kt_; i++){
            wtmp[i] = -conj(S.retptr(i, tstp)[0]);
            int kt2 = (i==-1 || i>kt_ ? kt_ : i-1);
            cntr::extrapolate_timestep(i-1,f3,kt2); 
            }
     cntr::response_integrate(tstp,h_,cc_m,wtmp,1,0,f3.ptr(0),1,0,integration::I<double>(kt_));
     cc_d += cc_m;
     double shift=(lattice_model_.U0_[tstp+1]+(8.0*lattice_model_.V_[tstp+1]));
     At1(0,0) = (2.0*cc_d);
     tmp1(0,0)=(shift*f3.ptr(tstp)[0]);
     At(0,0) = At1(0,0) + tmp1(0,0);
     delete[] wtmp;
}


    f.set_value(tstp,At);

}

template<class LATT>
void selfconsistency_pm<LATT>::clear_data(){

 

  for(int k=0;k<nk1_;k++){
    
     if(mpi_pid_kk_[k]==mpi_pid_){
    
     kk_functions_[k].Gk_dual.clear();
     kk_functions_[k].Gk_tilde.clear();
     kk_functions_[k].Wk_dual_c.clear();
     kk_functions_[k].Wk_dual_z.clear();
     kk_functions_[k].Wk_bare_c.clear();
     kk_functions_[k].Wk_bare_z.clear();
     kk_functions_[k].Gk_DMFT.clear();
    // kk_functions_[k].Gk_tilde.clear();
     kk_functions_[k].Convo_1temp.clear();
     kk_functions_[k].Convo_1temp_cc.clear();
     kk_functions_[k].Convo_2temp.clear();
     kk_functions_[k].Convo_2temp_cc.clear();
     kk_functions_[k].Convo_3temp.clear();
     kk_functions_[k].Convo_3temp_cc.clear();
     kk_functions_[k].Pi_mix.clear();
     kk_functions_[k].Sigma_dual_mix.clear();
    }


  }
   gtmp2_.clear();
   glatt_.clear();
   gk_all_timesteps_.clear();
   wk_c_all_timesteps_.clear();
   wk_z_all_timesteps_.clear();
   

   if(mpi_pid_==mpi_imp_){
         Gloc_up.clear();
         G1_.clear();
         G1c_.clear();
         G2_.clear();
         Delta_up.clear(); 
         Delta_up_cc.clear(); 
         G_Delta.clear();
         Delta_G.clear();
         Chi_c_.clear(); 
         Chi_z_.clear();
         retarded_w_charge_.clear(); 
      } 

}
template<class LATT>
void selfconsistency_pm<LATT>::get_dgw(){


 FILE *out;
 if(mpi_pid_ == mpi_imp_) out=fopen("obs_dual.out","w");
 for(int tstp1=-1;tstp1<=nt_;tstp1++){
 double dgw_occ=get_dual_local_dens(tstp1);
 double dgw_bare=get_dual_local_bare_dens(tstp1);
 if(mpi_pid_ == mpi_imp_){
  fprintf(out,"t: %d ",tstp1);
  fprintf(out," dens1: %.12g",dgw_bare);
  fprintf(out," dens1: %.12g",dgw_occ);
  fprintf(out,"\n");
   }
 }
 if(mpi_pid_ == mpi_imp_) fclose(out);

// for(int k=0;k<nk1_;k++){
//     if(mpi_pid_kk_[k]==mpi_pid_){
//     char filename1[1024],filename2[1024],filename3[1024];
//     std::sprintf(filename1,"DATA_check/sigma_sig_s_%u.out",k);
//     std::sprintf(filename2,"DATA_check/sigma_Hartree_%u.out",k);
//     std::sprintf(filename3,"DATA_check/sigma_sig_d_%u.out",k);
//     kk_functions_[k].Sigma_sg_.print_to_file(filename1);
//     kk_functions_[k].Hartree_.print_to_file(filename2);
//     kk_functions_[k].Sigma_sd_.print_to_file(filename3);    
//     }
//   }
 
//  if(mpi_pid_ == mpi_imp_){ 
//  ret_U_.print_to_file("retarded_U1.out");
//  tv_G_s_.print_to_file("retarded_U2.out");
//  }
//**/
}


template<class LATT>
void selfconsistency_pm<LATT>::init_for_observables(){
  
       // int nk_=lattice_model_.nk_;

  
       if(mpi_pid_==mpi_imp_){
        chi_lc = green1x1(nt_,ntau_,1,1);
        chi_lz = green1x1(nt_,ntau_,1,1);
        G_l = green1x1(nt_,ntau_,1,-1);
//        chi_bare_c = green1x1(nt_,ntau_,1,1);
//        chi_bare_z = green1x1(nt_,ntau_,1,1);
	sigma_corr = green1x1(nt_,ntau_,1,-1);
//        loc_pi_c=green1x1(nt_,ntau_,1,1);
//        loc_pi_lat_c=green1x1(nt_,ntau_,1,1);
//        loc_pi_z=green1x1(nt_,ntau_,1,1);
//        loc_pi_lat_z=green1x1(nt_,ntau_,1,1);         
        }
     
        for(int q=0;q<nk1_;q++){
            if(mpi_pid_kk_[q]==mpi_pid_){
               kk_functions_[q].init_observables();
             }
       }
 
}


template<class LATT>
void selfconsistency_pm<LATT>::get_real_G_r(int tstp, int &kt1, cntr::function<double> &U_){

     for(int k=0;k<nk1_;k++){
        if(mpi_pid_kk_[k]==mpi_pid_){
          kk_functions_[k].Get_real_Glatt(tstp,kt1,g_loc_,Delta_);     
          kk_functions_[k].Get_real_chi_latt(tstp,kt1,Pi_c_,Pi_z_,U_);
          kk_functions_[k].Get_sigma_correction(tstp,kt1,g_loc_);
         }
      }   
}
template<class LATT>
void selfconsistency_pm<LATT>::get_T_matrix_energy(cntr::herm_matrix<double> &Sigma_T,CFUNC &Sigma_H){
  
  int n;
  GREEN dg(nt_,ntau_,1,-1),dg_cc(nt_,ntau_,1,-1); 
  GREEN gD(nt_,ntau_,1,-1),Dg(nt_,ntau_,1,-1);
  CFUNC tmpG(nt_,1),tmpQ(nt_,1),tmpF(nt_,1);
  GREEN F(nt_,ntau_,1,-1),F_cc(nt_,ntau_,1,-1);
  GREEN dF_cc(nt_,ntau_,1,-1),DF(nt_,ntau_,1,-1);
  GREEN F_H(nt_,ntau_,1,-1),F_Q(nt_,ntau_,1,-1);

  for(n=-1;n<=nt_;n++){
      cntr::convolution_timestep_new(n,Dg,Delta_,g_loc_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,gD,g_loc_,Delta_,integration::I<double>(kt_),beta_,h_);
      cntr::deriv1_timestep(n,dg,g_loc_,g_loc_,beta_,h_,kt_);
      cntr::deriv2_timestep(n,dg_cc,g_loc_,g_loc_,beta_,h_,kt_);
    }

   
   for(n=-1;n<=nt_;n++){
      cdmatrix At(1,1);
      GREEN_TSTP tmp3(n,ntau_,1,-1),tmp4(n,ntau_,1,-1);
      tmp3.set_timestep(n,dg);
      tmp3.incr(Dg,-1.0);
      tmp4.set_timestep(n,dg_cc);
      tmp4.incr(gD,-1.0);
      F.set_timestep(n,tmp3);
      F_cc.set_timestep(n,tmp4); 
      if(n==-1){
      cdouble zz,ll,kk;
      tmp3.get_mat(0,ll);
      tmp3.get_mat(ntau_,kk);
      zz=-(ll+kk);
      At(0,0)=zz;
      Sigma_H.set_value(n,At);
      }else{
      cdouble xx,yy,zz;
      F.get_les(n,n,xx);
      F.get_gtr(n,n,yy);
      At(0,0)=(II*(yy-xx));
      Sigma_H.set_value(n,At);
      }
    }

//   Sigma_H.print_to_file("Sigma_H.out");

   for(n=-1;n<=nt_;n++){
      cntr::convolution_timestep_new(n,DF,Delta_,Delta_,F_cc,F,integration::I<double>(kt_),beta_,h_);
      cntr::deriv1_timestep(n,dF_cc,F_cc,F,beta_,h_,kt_);
      F_H.set_timestep(n,F); 
      F_H.right_multiply(n,Sigma_H,1.0);
   }
  
    for(n=-1;n<=nt_;n++){
         GREEN_TSTP tmp7(n,ntau_,1,-1);
         tmp7.set_timestep(n,dF_cc);
         tmp7.incr(DF,-1.0);
         tmp7.incr(F_H,-1.0);
         F_Q.set_timestep(n,tmp7);
    }

//    F.print_to_file("F.out");

    for(n=-1;n<=nt_;n++){

     cntr::vie2_timestep_sin(n,Sigma_T,tmpG,F,F_cc,tmpF,F_Q,tmpQ,beta_,h_,kt_);     

     }

}
/**
template<class LATT>
void selfconsistency_pm<LATT>::get_real_G_chi(int tstp, int &kt1, cntr::function<double> &U_){

       for(int k=0;k<nk1_;k++){
        if(mpi_pid_kk_[k]==mpi_pid_){
          kk_functions_[k].Get_real_chi_latt(tstp,kt1,Pi_c_,Pi_z_,U_);
     }
 }

}

template<class LATT>
void selfconsistency_pm<LATT>::get_sigma_bar(int tstp, int &kt1){

       for(int k=0;k<nk1_;k++){
        if(mpi_pid_kk_[k]==mpi_pid_){
          kk_functions_[k].Get_sigma_correction(tstp,kt1,g_loc_);
      }
   }

}
**/             
template<class LATT>
double selfconsistency_pm<LATT>::get_potential_energy(int tstp){

   cdouble cc_m(1,1);
   GREEN_TSTP gtmp1(tstp,ntau_,1,-1);
   double nktmp=0.0,nk=0.0;
   for(int k=0;k<lattice_model_.nk_bz_;k++){
    double wt=lattice_model_.kweight_bz_[k];
    int kk1=lattice_model_.idx_kk_[k];
    if(mpi_pid_kk_[kk1]==mpi_pid_){
     cntr::herm_matrix_timestep_view<double> tview1(tstp,gtmp1);
     tview1.incr_timestep(kk_functions_[kk1].sigma_bar,CPLX(wt,0.0));
    cntr::convolution_density_matrix(tstp,&cc_m,kk_functions_[kk1].sigma_bar,
kk_functions_[kk1].Gk_r,integration::I<double>(kt_),beta_,h_); 
     nktmp+=(std::real(cc_m)*wt);
   }
 }

  gtmp1.Reduce_timestep(tstp,mpi_imp_);  

  MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
  

 if(mpi_pid_==mpi_imp_) { 
  sigma_corr.set_timestep(tstp,gtmp1);
 // sigma_corr.print_to_file("sigma_corr.out");
  }

 return nk;

}




template<class LATT>
double selfconsistency_pm<LATT>::get_potential_singular(int tstp){
  
   cdmatrix At1(1,1),At2(1,1);
   At1.setZero();
   At2.setZero();
   double nktmp=0.0,nk=0.0;
   for(int k=0;k<lattice_model_.nk_bz_;k++){
    double wt=lattice_model_.kweight_bz_[k];
    int kk1=lattice_model_.idx_kk_[k];
    if(mpi_pid_kk_[kk1]==mpi_pid_){
    kk_functions_[kk1].Sigma_sg_.get_value(tstp,At1);
    kk_functions_[kk1].Gk_r.density_matrix(tstp,At2);
    nktmp+=(std::real((At1*At2).trace())*wt);
   }
 }

  MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
  return nk;

}
template<class LATT>
double selfconsistency_pm<LATT>::get_double_occupancy(int tstp){

   GREEN_TSTP gtmp1(tstp,ntau_,1,1);
   GREEN_TSTP gtmp2(tstp,ntau_,1,1);
   GREEN_TSTP gtmp3(tstp,ntau_,1,-1);
  // GREEN_TSTP gtmp4(tstp,ntau_,1,1);
  // GREEN_TSTP gtmp5(tstp,ntau_,1,1);
  // GREEN_TSTP gtmp6(tstp,ntau_,1,1);
  // GREEN_TSTP gtmp7(tstp,ntau_,1,1);

   
      for(int kk=0;kk<lattice_model_.nk_bz_;kk++){
        double wt=lattice_model_.kweight_bz_[kk];
         int kk1=lattice_model_.idx_kk_[kk];
         if(mpi_pid_kk_[kk1]==mpi_pid_){
          cntr::herm_matrix_timestep_view<double> tview1(tstp,gtmp1);
          cntr::herm_matrix_timestep_view<double> tview2(tstp,gtmp2);
          cntr::herm_matrix_timestep_view<double> tview3(tstp,gtmp3);
          tview1.incr_timestep(kk_functions_[kk1].Chi_rc,CPLX(wt,0.0));
          tview2.incr_timestep(kk_functions_[kk1].Chi_rz,CPLX(wt,0.0));
//          tview1.incr_timestep(kk_functions_[kk1].Pi_bar_c,CPLX(wt,0.0));
//          tview2.incr_timestep(kk_functions_[kk1].Pi_bar_z,CPLX(wt,0.0));
          tview3.incr_timestep(kk_functions_[kk1].Gk_r,CPLX(wt,0.0));
    //      cntr::herm_matrix_timestep_view<double> tview4(tstp,gtmp4);
    //      cntr::herm_matrix_timestep_view<double> tview5(tstp,gtmp5);
    //      cntr::herm_matrix_timestep_view<double> tview6(tstp,gtmp6);
    //      cntr::herm_matrix_timestep_view<double> tview7(tstp,gtmp7);
    //      tview4.incr_timestep(kk_functions_[kk1].Pi_bar_c,CPLX(wt,0.0));
    //      tview5.incr_timestep(kk_functions_[kk1].Pi_latt_c,CPLX(wt,0.0));
    //      tview6.incr_timestep(kk_functions_[kk1].Pi_bar_z,CPLX(wt,0.0));
    //      tview7.incr_timestep(kk_functions_[kk1].Pi_latt_z,CPLX(wt,0.0));
          }
       }
    // add all to task tid
//    #if CDMFT_CAN_USE_MPI==1
    gtmp1.Reduce_timestep(tstp,mpi_imp_);
    gtmp2.Reduce_timestep(tstp,mpi_imp_);
    gtmp3.Reduce_timestep(tstp,mpi_imp_);
  //  gtmp4.Reduce_timestep(tstp,mpi_imp_);
  //  gtmp5.Reduce_timestep(tstp,mpi_imp_);
  //  gtmp6.Reduce_timestep(tstp,mpi_imp_);
  //  gtmp7.Reduce_timestep(tstp,mpi_imp_);

//    #endif
    double docc;
    if(mpi_pid_==mpi_imp_){
     chi_lc.set_timestep(tstp,gtmp1);
     chi_lz.set_timestep(tstp,gtmp2);
     G_l.set_timestep(tstp,gtmp3);
   //  loc_pi_c.set_timestep(tstp,gtmp4);
   //  loc_pi_lat_c.set_timestep(tstp,gtmp5);
   //  loc_pi_z.set_timestep(tstp,gtmp6);
   //  loc_pi_lat_z.set_timestep(tstp,gtmp7);
     cdmatrix nn(1,1),ss(1,1),den(1,1),docc1(1,1);
     chi_lc.density_matrix(tstp,nn);
     chi_lz.density_matrix(tstp,ss);
     G_l.density_matrix(tstp,den);
 
     //std::cout << nn << ss << den << std::endl;   
 
     docc1 = -0.25*(-nn + ss - (4*den*den)); // Sign change due to density matrix...
      
     docc = docc1(0,0).real();
 
    }

    MPI::COMM_WORLD.Bcast(&docc,1,MPI::DOUBLE,mpi_imp_);

    return docc;
}

template<class LATT>
double selfconsistency_pm<LATT>::get_double_DMFT(int tstp){

 double docc2;
 if(mpi_pid_==mpi_imp_){
 cdmatrix nn(1,1),ss(1,1),den(1,1),docc1(1,1);
 Chi_c_.density_matrix(tstp,nn);
 Chi_z_.density_matrix(tstp,ss);
 g_loc_.density_matrix(tstp,den);

 docc1 = -0.25*(-nn + ss - (4*den*den));

 docc2 = docc1(0,0).real();
 }

 MPI::COMM_WORLD.Bcast(&docc2,1,MPI::DOUBLE,mpi_imp_);

 return docc2;

}

template<class LATT>
void selfconsistency_pm<LATT>::get_local_dual_G(int tstp){

   GREEN_TSTP gtmp1(tstp,ntau_,1,-1);
   GREEN_TSTP gtmp2(tstp,ntau_,1,-1);
//   GREEN_TSTP gtmp3(tstp,ntau_,1,1);
//   GREEN_TSTP gtmp4(tstp,ntau_,1,1);
      for(int kk=0;kk<lattice_model_.nk_bz_;kk++){
        double wt=lattice_model_.kweight_bz_[kk];
         int kk1=lattice_model_.idx_kk_[kk];
         if(mpi_pid_kk_[kk1]==mpi_pid_){
          cntr::herm_matrix_timestep_view<double> tview1(tstp,gtmp1);
          cntr::herm_matrix_timestep_view<double> tview2(tstp,gtmp2);
	  tview1.incr_timestep(kk_functions_[kk1].Gk_tilde,CPLX(wt,0.0));
	  tview2.incr_timestep(kk_functions_[kk1].Gk_dual,CPLX(wt,0.0));
	 }
      } 
//         #if CDMFT_CAN_USE_MPI==1
          gtmp1.Reduce_timestep(tstp,mpi_imp_);
          gtmp2.Reduce_timestep(tstp,mpi_imp_);
//         #endif

  if(mpi_pid_==mpi_imp_){
   G_tilde_local.set_timestep(tstp,gtmp1);
   G_dual_local.set_timestep(tstp,gtmp2);
 //  G_tilde_local.print_to_file("G_tilde_local.out");
 //  G_dual_local.print_to_file("G_dual_local.out");
  }

}


template<class LATT>
void selfconsistency_pm<LATT>::get_local_bare(int tstp){

   GREEN_TSTP gtmp1(tstp,ntau_,1,1);
   GREEN_TSTP gtmp2(tstp,ntau_,1,1);
   GREEN_TSTP gtmp3(tstp,ntau_,1,1);
   GREEN_TSTP gtmp4(tstp,ntau_,1,1);
      for(int kk=0;kk<lattice_model_.nk_bz_;kk++){
        double wt=lattice_model_.kweight_bz_[kk];
         int kk1=lattice_model_.idx_kk_[kk];
         if(mpi_pid_kk_[kk1]==mpi_pid_){
          cntr::herm_matrix_timestep_view<double> tview1(tstp,gtmp1);
          cntr::herm_matrix_timestep_view<double> tview2(tstp,gtmp2);
          cntr::herm_matrix_timestep_view<double> tview3(tstp,gtmp3);
          cntr::herm_matrix_timestep_view<double> tview4(tstp,gtmp4);
          tview1.set_timestep(tstp,kk_functions_[kk1].Chi_rc);
          tview2.set_timestep(tstp,kk_functions_[kk1].Chi_rz);
          gtmp1.left_multiply(tstp,kk_functions_[kk1].Vertex_c);
          gtmp1.right_multiply(tstp,kk_functions_[kk1].Vertex_c);
          gtmp2.left_multiply(tstp,kk_functions_[kk1].Vertex_z);
          gtmp2.right_multiply(tstp,kk_functions_[kk1].Vertex_z); 
//          kk_functions_[kk1].sc_c.set_timestep(tstp,gtmp1);
//          kk_functions_[kk1].sc_z.set_timestep(tstp,gtmp2);
          tview3.incr_timestep(tview1,CPLX(wt,0.0));
          tview4.incr_timestep(tview2,CPLX(wt,0.0));
          }
       }
    // add all to task tid
//    #if CDMFT_CAN_USE_MPI==1
    gtmp3.Reduce_timestep(tstp,mpi_imp_);
    gtmp4.Reduce_timestep(tstp,mpi_imp_);
//    #endif

// if(mpi_pid_==mpi_imp_){
//   chi_bare_c.set_timestep(tstp,gtmp3);
//   chi_bare_z.set_timestep(tstp,gtmp4);
//   chi_bare_c.print_to_file("chi_bare_c.dat");
//   chi_bare_z.print_to_file("chi_bare_z.dat")
// }
/**/
}
template<class LATT>
void selfconsistency_pm<LATT>::Get_matsubara(){

  cdmatrix At1(1,1),At2(1,1),At3(1,1),At4(1,1);
  for(int k=0;k<nk1_;k++){
     if(mpi_pid_kk_[k]==mpi_pid_){
      if((lattice_model_.kpoints_[k](0)==PI)&&(lattice_model_.kpoints_[k](1)==PI)){
     double dtau_= beta_ / ntau_;
     At1.setZero();
     At2.setZero();
     At3.setZero();
     At4.setZero();
//     At3.setZero();
     cdmatrix tmp1(1,1),tmp2(1,1),tmp3(1,1),tmp4(1,1);
     for(int i=0;i<=ntau_;i++){
    //**kk_functions_[k].Chi_rc.get_mat(i,tmp1);
    //** kk_functions_[k].Chi_rz.get_mat(i,tmp2);
     kk_functions_[k].Pi_latt_c.get_mat(i,tmp1);
     kk_functions_[k].Pi_latt_z.get_mat(i,tmp2);
     Pi_c_.get_mat(i,tmp3);
     Pi_z_.get_mat(i,tmp4);	     
//     kk_functions_[k].Pi_dual.get_mat(i,tmp1);
//     kk_functions_[k].Sigma_dual.get_mat(i,tmp2);
//     kk_functions_[k].Gk_r.get_mat(i,tmp3);
//     tmp4(0,0)=CPLX(cos((PI*i)/ntau_),sin((PI*i)/ntau_));
     At1+= (integration::I<double>(kt_).gregory_weights(ntau_,i)*tmp1);
     At2+= (integration::I<double>(kt_).gregory_weights(ntau_,i)*tmp2);
     At3+= (integration::I<double>(kt_).gregory_weights(ntau_,i)*tmp3);
     At4+= (integration::I<double>(kt_).gregory_weights(ntau_,i)*tmp4);

//     At3+= integration::I<double>(kt_).gregory_weights(ntau_,i)*tmp3*tmp4;
    }
   
   At1(0,0)*=(dtau_);
   At2(0,0)*=(dtau_);
   At3(0,0)*=(dtau_);
   At4(0,0)*=(dtau_);


//   std::cout << At1(0,0) << At2(0,0);

    FILE *out1;
    out1=fopen("charge_spin_pi_pi.out","w");
    fprintf(out1," %5.6f ",1.0/beta_);
    fprintf(out1," %5.6f ",((lattice_model_.U0_[0]*0.5)-(4.0*lattice_model_.V_[0])));
    //fprintf(out1," %5.6f ",lattice_model_.V_[0]);
    fprintf(out1," %5.6f ",1.0/At1(0,0).real());
//    fprintf(out1," %5.6f ",At1(0,0).imag());
    fprintf(out1," %5.6f ",1.0/At2(0,0).real());
    fprintf(out1," %5.6f ",1.0/At3(0,0).real());
    fprintf(out1," %5.6f ",1.0/At4(0,0).real());
//    fprintf(out1," %5.6f ",At2(0,0).imag());
    fprintf(out1,"\n");
    fclose(out1);

    }

   }
  
  }
}
/**/
template<class LATT>
void selfconsistency_pm<LATT>::print_DMFT_file(){

 for(int k=0;k<nk1_;k++){
     if(mpi_pid_kk_[k]==mpi_pid_){
     char filename[1024];
     if(lattice_model_.kpoints_[k](1)==0){
     std::sprintf(filename,"DATA_kx_0/G_DMFT_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Gk_DMFT.print_to_file(filename);
     }
     else if(lattice_model_.kpoints_[k](0)==PI){
     std::sprintf(filename,"DATA_0_ky/G_DMFT_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_DMFT.print_to_file(filename);
     }
    else if(lattice_model_.kpoints_[k](0)==lattice_model_.kpoints_[k](1)){
     std::sprintf(filename,"DATA_kx_ky/G_DMFT_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_DMFT.print_to_file(filename);

   }

  }

 }


}

template<class LATT>
void selfconsistency_pm<LATT>::print_DGW(){

 for(int k=0;k<nk1_;k++){
     if(mpi_pid_kk_[k]==mpi_pid_){
     char filename[1024];
     if(lattice_model_.kpoints_[k](1)==0){
     std::sprintf(filename,"DATA_kx_0/G_dual_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Gk_dual.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_0/G_tilde_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Gk_tilde.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Wk_dual_c_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Wk_dual_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Wk_bare_c_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Wk_bare_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Wk_bare_z_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Wk_bare_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Wk_dual_z_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Wk_bare_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Sigma_dual_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Sigma_dual.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_0/Pi_dual_%g.out",lattice_model_.kpoints_[k](0));
//     kk_functions_[k].Pi_dual.print_to_file(filename);


     }
     else if(lattice_model_.kpoints_[k](0)==PI){
     std::sprintf(filename,"DATA_0_ky/G_dual_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_dual.print_to_file(filename);
     std::sprintf(filename,"DATA_0_ky/G_tilde_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_tilde.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Wk_dual_c_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_dual_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Wk_bare_c_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_bare_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Wk_bare_z_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_bare_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Wk_dual_z_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_bare_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Sigma_dual_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Sigma_dual.print_to_file(filename);
//     std::sprintf(filename,"DATA_0_ky/Pi_dual_%g.out",lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Pi_dual.print_to_file(filename);

     }
    else if(lattice_model_.kpoints_[k](0)==lattice_model_.kpoints_[k](1)){
     std::sprintf(filename,"DATA_kx_ky/G_dual_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_dual.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_ky/G_tilde_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_tilde.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Wk_dual_c_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_dual_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Wk_bare_c_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_bare_c.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Wk_dual_z_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_dual_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Wk_bare_z_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Wk_bare_z.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Sigma_dual_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Sigma_dual.print_to_file(filename);
//     std::sprintf(filename,"DATA_kx_ky/Pi_dual_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
//     kk_functions_[k].Pi_dual.print_to_file(filename);


   }

  }

 }


}


/**/
template<class LATT>
void selfconsistency_pm<LATT>::print_file(){

 for(int k=0;k<nk1_;k++){
     if(mpi_pid_kk_[k]==mpi_pid_){   
     char filename[1024];
     if(lattice_model_.kpoints_[k](1)==0){
     std::sprintf(filename,"DATA_kx_0/Chi_c_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Chi_rc.print_to_file(filename);
//       kk_functions_[k].Pi_dual.print_to_file(filename);
//     kk_functions_[k].Pi_latt_c.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_0/Chi_z_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Chi_rz.print_to_file(filename);
//     kk_functions_[k].Pi_dual.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_0/G_r_%g.out",lattice_model_.kpoints_[k](0));
     kk_functions_[k].Gk_r.print_to_file(filename);
//     kk_functions_[k].Gk_dual.print_to_file(filename);
     }
     else if(lattice_model_.kpoints_[k](0)==PI){
     std::sprintf(filename,"DATA_0_ky/Chi_c_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Chi_rc.print_to_file(filename);
//       kk_functions_[k].Pi_dual.print_to_file(filename);     
     std::sprintf(filename,"DATA_0_ky/Chi_z_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Chi_rz.print_to_file(filename);
//     kk_functions_[k].Pi_bar_z.print_to_file(filename);
     std::sprintf(filename,"DATA_0_ky/G_r_%g.out",lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_r.print_to_file(filename);
//     kk_functions_[k].Gk_dual.print_to_file(filename);
     }
    else if(lattice_model_.kpoints_[k](0)==lattice_model_.kpoints_[k](1)){
     std::sprintf(filename,"DATA_kx_ky/Chi_c_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Chi_rc.print_to_file(filename);
//       kk_functions_[k].Pi_dual.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_ky/Chi_z_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Chi_rz.print_to_file(filename);
//     kk_functions_[k].Pi_bar_z.print_to_file(filename);
     std::sprintf(filename,"DATA_kx_ky/G_r_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
     kk_functions_[k].Gk_r.print_to_file(filename);
//     kk_functions_[k].Gk_dual.print_to_file(filename);

   }
     
  }

 }

//  if(mpi_pid_==mpi_imp_){
  
//  chi_lc.print_to_file("chi_c_loc.out");
  
//  chi_lz.print_to_file("chi_z_loc.out"); 
   
//  G_l.print_to_file("G_r_loc.out");  


//  }
}

template<class LATT>
void selfconsistency_pm<LATT>::print_sigma_file(){

 for(int k=0;k<nk1_;k++){
     if(mpi_pid_kk_[k]==mpi_pid_){
     char filename[1024];
     if(lattice_model_.kpoints_[k](1)==0){
      std::sprintf(filename,"DATA_kx_0/sigma_bar_%g.out",lattice_model_.kpoints_[k](0));
      kk_functions_[k].sigma_bar.print_to_file(filename);
      }
     else if(lattice_model_.kpoints_[k](0)==PI){
       std::sprintf(filename,"DATA_0_ky/sigma_bar_%g.out",lattice_model_.kpoints_[k](1));
       kk_functions_[k].sigma_bar.print_to_file(filename);
     } 
     else if(lattice_model_.kpoints_[k](0)==lattice_model_.kpoints_[k](1)){
       std::sprintf(filename,"DATA_kx_ky/sigma_bar_%g_%g.out",lattice_model_.kpoints_[k](0),lattice_model_.kpoints_[k](1));
       kk_functions_[k].sigma_bar.print_to_file(filename);

    }
   
//     std::sprintf(filename,"DATA_sigma/Sigma_dual_%u.out",k);
//     kk_functions_[k].Sigma_bar.print_to_file(filename);


   }

  }

 // if(mpi_pid_==mpi_imp_) Sigma_T.print_to_file("Sigma_DMFT.out");

 

}

/**/
template<class LATT>
void selfconsistency_pm<LATT>::Get_G_dual_bare(int tstp,cntr::herm_matrix<double> &S, cntr::herm_matrix<double> &P){


      S.set_timestep_zero(tstp);
      S.set_timestep(tstp,P);
      S.incr_timestep(tstp,g_loc_,CPLX(-1.0, 0.0));
}

template<class LATT>
void selfconsistency_pm<LATT>::Get_W_dual_bare(int tstp, cntr::function<double> &f_c, cntr::function<double> &f_z, 
                             cntr::herm_matrix<double> &W_c, cntr::herm_matrix<double> &W_z){

      int n; 
//      int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);
      int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);
//      int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);

     GREEN g_temp1(n2,ntau_,1,1),g_temp2(n2,ntau_,1,1),g_temp1_cc(n2,ntau_,1,1),
     g_temp2_cc(n2,ntau_,1,1),g_temp3(n2,ntau_,1,1), g_temp4(n2,ntau_,1,1);
//     g_temp3_cc(n2,ntau_,1,1),g_temp4_cc(n2,ntau_,1,1);

//      GREEN g_temp1(nt_,ntau_,1,1),g_temp2(nt_,ntau_,1,1),g_temp1_cc(nt_,ntau_,1,1),
//     g_temp2_cc(nt_,ntau_,1,1),g_temp3(nt_,ntau_,1,1), g_temp4(nt_,ntau_,1,1);


      for(n=-1;n<=n2;n++){
         
         g_temp1.set_timestep(n,Pi_c_);
         g_temp1_cc.set_timestep(n,Pi_c_);

         g_temp2.set_timestep(n,Pi_z_);
         g_temp2_cc.set_timestep(n,Pi_z_);


         g_temp1.left_multiply(n,f_c,-1.0);
         g_temp1_cc.right_multiply(n,f_c,-1.0);

         g_temp2.left_multiply(n,f_z,-1.0);
         g_temp2_cc.right_multiply(n,f_z,-1.0);

         g_temp3.set_timestep(n,g_temp1);
//         g_temp3_cc.set_timestep(n,g_temp1_cc);
         g_temp4.set_timestep(n,g_temp2);
//         g_temp4_cc.set_timestep(n,g_temp2_cc);

	 g_temp3.right_multiply(n,f_c,-1.0);
//         g_temp3_cc.right_multiply(n,f_c,-1.0);

         g_temp4.right_multiply(n,f_z,-1.0);
//         g_temp4_cc.right_multiply(n,f_z,-1.0);


        }

//       cntr::function<double> tmpF1sin(nt_,1), tmpF2sin(nt_,1), tmpQ1sin(nt_,1), 
//                  tmpQ2sin(nt_,1), tmpG1sin(nt_,1), tmpG2sin(nt_,1);

//         tmpF1sin.set_zero();
//         tmpQ1sin.set_zero();
//         tmpQ2sin.set_zero();
//         tmpQ1sin.set_matrixelement(0,0,f_c,0,0);
//         tmpG1sin.set_zero();

//        for(n=-1;n<=n2;n++){
//         cdmatrix At1(1,1),At2(1,1);
//         f_c.get_value(n,At1);
//         f_z.get_value(n,At2);
//         tmpQ1sin.set_value(n,At1);
//         tmpQ2sin.set_value(n,At2);
//        }
//        tmpQsin.set_matrixelement(0,0,Sigma_sg_,0,0);
//                tmpGsin.set_zero();
//

         if(tstp==-1){
                cntr::vie2_mat_fixpoint(W_c,g_temp1,g_temp1_cc,g_temp3,beta_,integration::I<double>(kt_),10);
                cntr::vie2_mat_fixpoint(W_z,g_temp2,g_temp2_cc,g_temp4,beta_,integration::I<double>(kt_),10);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(W_c);
                cntr::set_t0_from_mat(W_z);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(W_c,g_temp1,g_temp1_cc,g_temp3,integration::I<double>(kt_),beta_,h_);
               cntr::vie2_start(W_z,g_temp2,g_temp2_cc,g_temp4,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,W_c,g_temp1,g_temp1_cc,g_temp3,beta_,h_,kt_);
             cntr::vie2_timestep(tstp,W_z,g_temp2,g_temp2_cc,g_temp4,beta_,h_,kt_);

         }

      

//       cntr::vie2_timestep_sin(tstp,W_c,tmpG1sin,g_temp1,g_temp1_cc,tmpF1sin,g_temp3,tmpQ1sin,beta_,h_,kt_);

//       tmpF2sin.set_zero();
//       tmpQ2sin.set_matrixelement(0,0,f_z,0,0);
//       tmpG2sin.set_zero();


//       cntr::vie2_timestep_sin(tstp,W_z,tmpG2sin,g_temp2,g_temp2_cc,tmpF2sin,g_temp4,tmpQ2sin,beta_,h_,kt_);
    

}

template <class LATT>
void selfconsistency_pm<LATT>::get_Sigma_dual(int tstp,int kk,GREEN &S){
                // \Sigma_k (t,t') = i \sum_q [ G_{k-q} (t,t') { W^c_q (t,t') + 3 * W^z_q(t,t')}]  
                //\Sigma_k (t,t') = i \sum_q [ G_{k+q} (t,t') { W^c_q (t,t') + 3 * W^z_q(t,t')}] //Zhenya derivation
                GREEN_TSTP stmp1(tstp,ntau_,1,-1);
                GREEN_TSTP stmp2(tstp,ntau_,1,-1);
                GREEN_TSTP stmp3(tstp,ntau_,1,-1);
                S.set_timestep_zero(tstp);
                int kq,kq1,qq;
                for(int q=0;q<lattice_model_.nk_bz_;q++){
                      double wk=lattice_model_.kweight_bz_[q];
                      stmp3.clear();
                 //     kq=lattice_model_.add_kpoints_idx(lattice_model_.idx_inv_[kk],q,-1);
                      kq=lattice_model_.add_kpoints_idx(lattice_model_.idx_inv_[kk],q,1);
                      kq1=lattice_model_.idx_kk_[kq];
                      qq=lattice_model_.idx_kk_[q];
            cntr::Bubble2(tstp,stmp1,0,0,gk_all_timesteps_.G()[kq1],gk_all_timesteps_.G()[kq1],0,0,
                                             wk_c_all_timesteps_.G()[qq],wk_c_all_timesteps_.G()[qq],0,0);
            cntr::Bubble2(tstp,stmp2,0,0,gk_all_timesteps_.G()[kq1],gk_all_timesteps_.G()[kq1],0,0,
                                             wk_z_all_timesteps_.G()[qq],wk_z_all_timesteps_.G()[qq],0,0);
                        stmp3.incr(stmp1,1.0);
                        stmp3.incr(stmp2,3.0);   //switch-off spin channel
                        S.incr_timestep(tstp,stmp3,wk);
                        
                }
               
 }


template <class LATT>
void selfconsistency_pm<LATT>::get_Sigma_Hartree(int tstp,int kk,CFUNC &S){
    // \Sigma^{singular}_k (t,t) = i \sum_q G_{k-q} (t,t) V_q,c (t) 
    // \Sigma^{singular}_k (t,t) = i \sum_q G_{k+q} (t,t) V_q,c (t) \\ Zhenya derivation
     cdmatrix stemp1(1,1),stemp2(1,1),stemp3(1,1),Z2(1,1);
     stemp1.setZero();
     int kq,kq1,qq;
     dvector tmp(2);
     cdmatrix rtmp(1,1),hktmp(1,1);
     for(int q=0;q<lattice_model_.nk_bz_;q++){
     double wk=lattice_model_.kweight_bz_[q];
 //    kq=lattice_model_.add_kpoints_idx(lattice_model_.idx_inv_[kk],q,-1);
     kq=lattice_model_.add_kpoints_idx(lattice_model_.idx_inv_[kk],q,1);
     kq1=lattice_model_.idx_kk_[kq];
     qq=lattice_model_.idx_kk_[q];
     gk_all_timesteps_.G()[kq1].density_matrix(tstp,rtmp);
     tmp(0)=lattice_model_.kpoints_[qq](0);
     tmp(1)=lattice_model_.kpoints_[qq](1);
     double Vertex=(2.0*lattice_model_.V_[tstp+1])*(cos(tmp(0))+cos(tmp(1)));
       stemp1(0,0)+=(-std::real(rtmp.trace())*wk*Vertex);
//     if(kk==0) std::cout << Vertex <<"\t" <<  tmp(0) <<"\t"<< tmp(1) << std::endl;   
//       stemp1(0,0)+=(-rtmp.trace()*wk*(Vertex+(0.75*lattice_model_.U0_[tstp+1]))); // c only
//       stemp1(0,0)+=(rtmp.trace()*wk*(0.25*lattice_model_.U0_[tstp+1])); // s only
     }   

     S.set_value(tstp,stemp1);

  }


template <class LATT>
void selfconsistency_pm<LATT>::gather_Gk_dual_timestep(int tstp){
                gk_all_timesteps_.reset_tstp(tstp);
                // read Gk_ to timestep
                for(int k=0;k<nk1_;k++){
                  if(mpi_pid_kk_[k]==mpi_pid_){
                gk_all_timesteps_.G()[k].get_data(kk_functions_[k].Gk_dual);
                }
              }
                // distribute to all nodes
                gk_all_timesteps_.mpi_bcast_all();
}
/**/
template <class LATT>
void selfconsistency_pm<LATT>::Get_latt_bubble(int tstp,int qq,GREEN &P){
              // Pi_q = -2*i*\sum_k G_k (t,t') G_{k-q} (t',t)
              // Pi_q = -2*i*\sum_k G_k (t,t') G_{k+q} (t',t) \\ Zhenya derivation   
                int kq,kq1,kk1;
                GREEN_TSTP ptmp(tstp,ntau_,1,1);
                P.set_timestep_zero(tstp);
                for(int kk=0;kk<lattice_model_.nk_bz_;kk++){
                     double wk=lattice_model_.kweight_bz_[kk];
                    // kq=lattice_model_.add_kpoints_idx(kk,lattice_model_.idx_inv_[qq],-1);
                     kq=lattice_model_.add_kpoints_idx(kk,lattice_model_.idx_inv_[qq],1);
                     kq1=lattice_model_.idx_kk_[kq];
                     kk1=lattice_model_.idx_kk_[kk];
  cntr::Bubble1(tstp,ptmp,0,0,gk_all_timesteps_.G()[kk1],gk_all_timesteps_.G()[kk1],
                     0,0,gk_all_timesteps_.G()[kq1],gk_all_timesteps_.G()[kq1],0,0);
               //    P.incr_timestep(tstp,ptmp,CPLX(-2.0*wk,0.0)); // spin degeneracy (2) is included 
                   P.incr_timestep(tstp,ptmp,-2.0*wk);
                }
        }

template <class LATT>
void selfconsistency_pm<LATT>::gather_Wk_dual_timestep(int tstp){
           wk_c_all_timesteps_.reset_tstp(tstp);
           wk_z_all_timesteps_.reset_tstp(tstp);
                // read Gk_ to timestep
           for(int q=0;q<nk1_;q++){
             if(mpi_pid_kk_[q]==mpi_pid_){ 
               wk_c_all_timesteps_.G()[q].get_data(kk_functions_[q].Wk_dual_c);
               wk_z_all_timesteps_.G()[q].get_data(kk_functions_[q].Wk_dual_z);
            } 
          // distribute to all nodes
          wk_c_all_timesteps_.mpi_bcast_all();
          wk_z_all_timesteps_.mpi_bcast_all();
       }

}
template <class LATT>
void selfconsistency_pm<LATT>::Get_initialize_zero(int tstp, cntr::herm_matrix<double> &S, 
            cntr::herm_matrix<double> &P, cntr::herm_matrix<double> &Q){
 
  S.set_timestep_zero(tstp);
  P.set_timestep_zero(tstp);
  Q.set_timestep_zero(tstp);

}

template <class LATT>
double selfconsistency_pm<LATT>::get_PE_chi_c(int tstp){
                cdmatrix rtmp(1,1),hktmp(1,1);
		dvector tmp(2);
                double nktmp=0.0,nk=0.0;
		//cdouble nktmp(0,0),nk(0,0);
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                            kk_functions_[kk1].Chi_rc.density_matrix(tstp,rtmp);
                            tmp(0)=lattice_model_.kpoints_[kk1](0);
                            tmp(1)=lattice_model_.kpoints_[kk1](1);
                            double Vertex=(2.0*lattice_model_.V_[tstp+1])*(cos(tmp(0))+cos(tmp(1)));
                            nktmp+=(std::real(rtmp.trace())*Vertex*wt);
                        }
                }
                MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
                //MPI_Allreduce(&nktmp,&nk,1,MPI_C_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
 		return nk;
}

/**/
template <class LATT>
double selfconsistency_pm<LATT>::get_ekin(int tstp){
                cdmatrix rtmp(1,1),hktmp(1,1);
                double ekintmp=0.0,ekin=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                            kk_functions_[kk1].Gk_DMFT.density_matrix(tstp,rtmp);
                            kk_functions_[kk1].hk_.get_value(tstp,hktmp);
                            ekintmp+=std::real((hktmp*rtmp).trace())*wt;
                        }
                }
//                ekin=ekintmp;
//             #if CDMFT_CAN_USE_MPI==1
//                    ekin=0.0;
//                    MPI::COMM_WORLD.Allreduce(&ekintmp,&ekin,1,MPI::DOUBLE,MPI_SUM);
             MPI_Allreduce(&ekintmp,&ekin,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//                #endif
             return ekin;
 }

template <class LATT>
double selfconsistency_pm<LATT>::get_dens(int tstp){
  cdmatrix rtmp(1,1),hktmp(1,1);
                double nktmp=0.0,nk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_DMFT.density_matrix(tstp,rtmp);
                                nktmp+=std::real((rtmp).trace())*wt;
                        }
                }
//               nk=nktmp;
//               #if CDMFT_CAN_USE_MPI==1
//               nk=0.0;
//               MPI::COMM_WORLD.Allreduce(&nktmp,&nk,1,MPI::DOUBLE,MPI_SUM);
               MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return nk;
 }
  
template <class LATT>
double selfconsistency_pm<LATT>::get_dual_local_dens(int tstp){
                cdmatrix rtmp(1,1);
                double nktmp=0.0,nk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                         if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_dual.density_matrix(tstp,rtmp);
                                nktmp+=std::real((rtmp).trace())*wt;
                    }
                 }
//               nk=nktmp;
//               #if CDMFT_CAN_USE_MPI==1
//               nk=0.0;
//               MPI::COMM_WORLD.Allreduce(&nktmp,&nk,1,MPI::DOUBLE,MPI_SUM);
               MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return nk;
} 

template <class LATT>
//cdouble selfconsistency_pm<LATT>::get_dual_local_G_les(int tstp){
double selfconsistency_pm<LATT>::get_dual_local_G_les(int tstp){
                cdmatrix rtmp(1,1);
//                At1.setZero();
//                cdouble nktmp(0,0),nk(0,0);
		double nktmp=0.0,nk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                         if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_dual.density_matrix(tstp,rtmp);
                                nktmp+=std::real((rtmp).trace())*wt;
                                //nktmp=At1(0,0);            
                                
                    }
                 }

             //MPI_Allreduce(&nktmp,&nk,1,MPI_C_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
	     MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
             return nk;
}

template <class LATT>
cdouble selfconsistency_pm<LATT>::get_dual_local_tv(int tstp){
                cdmatrix rtmp(1,1);
                cdouble nktmp(0.0),nk(0.0);
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                         if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_dual.get_tv(tstp,tstp,rtmp); //density_matrix(tstp,rtmp);
                                nktmp+=((rtmp.trace())*wt*CPLX(0,-1));
                    }
                 }
//               nk=nktmp;
//               #if CDMFT_CAN_USE_MPI==1
//               nk=0.0;
//               MPI::COMM_WORLD.Allreduce(&nktmp,&nk,1,MPI::DOUBLE,MPI_SUM);
//                 MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
                 MPI_Allreduce(&nktmp,&nk,1,MPI_C_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return nk;
}

template <class LATT>
double selfconsistency_pm<LATT>::get_dual_local_bare_dens(int tstp){
                cdmatrix rtmp(1,1);
                double nktmp=0.0,nk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                         if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_tilde.density_matrix(tstp,rtmp);
                                nktmp+=std::real((rtmp).trace())*wt;
                    }
                 }
//               nk=nktmp;
//               #if CDMFT_CAN_USE_MPI==1
//               nk=0.0;
//               MPI::COMM_WORLD.Allreduce(&nktmp,&nk,1,MPI::DOUBLE,MPI_SUM);
                 MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return nk;
}
template <class LATT>
double selfconsistency_pm<LATT>::get_current(int tstp){
  cdmatrix rtmp(1,1),hktmp(1,1);
                double vktmp=0.0,vk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_DMFT.density_matrix(tstp,rtmp);
                                lattice_model_.vk(hktmp,tstp,kk_functions_[kk1].kk_);
                                vktmp+=std::real((hktmp*rtmp).trace())*wt;
                        }
                }
//                vk=vktmp;
               // add all to task ti
//               #if CDMFT_CAN_USE_MPI==1
//               vk=0.0;
//               MPI::COMM_WORLD.Allreduce(&vktmp,&vk,1,MPI::DOUBLE,MPI_SUM);
                 MPI_Allreduce(&vktmp,&vk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return vk;
 }

template <class LATT>
void selfconsistency_pm<LATT>::get_imp_polarization(int tstp, cntr::function<double> &U_, 
          cntr::herm_matrix<double> &chi_c, cntr::herm_matrix<double> &chi_z){

    int n;
//    int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);
 
//    int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//    int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);
   

    int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);   


    GREEN g_temp1(n2,ntau_,1,1),g_temp2(n2,ntau_,1,1),
          g_temp1_cc(n2,ntau_,1,1),g_temp2_cc(n2,ntau_,1,1);

//   GREEN g_temp1(nt_,ntau_,1,1),g_temp2(nt_,ntau_,1,1),
//          g_temp1_cc(nt_,ntau_,1,1),g_temp2_cc(nt_,ntau_,1,1);

    for(n=-1;n<=n2;n++){
 
         g_temp1.set_timestep(n,chi_c);
         g_temp1_cc.set_timestep(n,chi_c);

         g_temp2.set_timestep(n,chi_z);
         g_temp2_cc.set_timestep(n,chi_z);

         g_temp1.right_multiply(n,U_,0.5);
         g_temp1_cc.left_multiply(n,U_,0.5);

         g_temp2.right_multiply(n,U_,-0.5);
         g_temp2_cc.left_multiply(n,U_,-0.5);


        }
      
        if(tstp==-1){
                cntr::vie2_mat_fixpoint(Pi_c_,g_temp1,g_temp1_cc,chi_c,beta_,integration::I<double>(kt_),10);
		cntr::vie2_mat_fixpoint(Pi_z_,g_temp2,g_temp2_cc,chi_z,beta_,integration::I<double>(kt_),10);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Pi_c_);
		cntr::set_t0_from_mat(Pi_z_);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Pi_c_,g_temp1,g_temp1_cc,chi_c,integration::I<double>(kt_),beta_,h_);
	       cntr::vie2_start(Pi_z_,g_temp2,g_temp2_cc,chi_z,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Pi_c_,g_temp1,g_temp1_cc,chi_c,beta_,h_,kt_);
	     cntr::vie2_timestep(tstp,Pi_z_,g_temp2,g_temp2_cc,chi_z,beta_,h_,kt_);

         }


//    cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);
//       tmpFsin.set_zero();
//       tmpQsin.set_zero();
//       tmpGsin.set_zero();
   


//       cntr::vie2_timestep_sin(tstp,Pi_c_,tmpGsin,g_temp1,g_temp1_cc,tmpFsin,chi_c,tmpQsin,beta_,h_,kt_);

//       tmpFsin.set_zero();
//       tmpQsin.set_zero();
//       tmpGsin.set_zero();


//       cntr::vie2_timestep_sin(tstp,Pi_z_,tmpGsin,g_temp2,g_temp2_cc,tmpFsin,chi_z,tmpQsin,beta_,h_,kt_); 
 }

template <class LATT>
double selfconsistency_pm<LATT>::get_Dgw_dens(int tstp){
                cdmatrix rtmp(1,1);
                double nktmp=0.0,nk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                         if(mpi_pid_kk_[kk1]==mpi_pid_){
                            kk_functions_[kk1].Gk_r.density_matrix(tstp,rtmp);
                             nktmp+=std::real((rtmp).trace())*wt;
               }
           }
//           nk=nktmp;
//           #if CDMFT_CAN_USE_MPI==1
//           nk=0.0;
//           MPI::COMM_WORLD.Allreduce(&nktmp,&nk,1,MPI::DOUBLE,MPI_SUM);
             MPI_Allreduce(&nktmp,&nk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//           #endif
           return nk;

}

template <class LATT>
double selfconsistency_pm<LATT>::get_average_disper(int tstp){
                cdmatrix hktmp(1,1);
                double ekintmp=0.0,ekin=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
//                            kk_functions_[kk1].Gk_r.density_matrix(tstp,rtmp);
                            kk_functions_[kk1].hk_.get_value(tstp,hktmp);
                            //ekintmp += wt;
                            ekintmp+=hktmp(0,0).real()*wt;
                        }
                }
               
               MPI_Allreduce(&ekintmp,&ekin,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
               return ekin;

}

template <class LATT>
double selfconsistency_pm<LATT>::get_Dgw_ekin(int tstp){
                cdmatrix rtmp(1,1),hktmp(1,1);
                double ekintmp=0.0,ekin=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                            kk_functions_[kk1].Gk_r.density_matrix(tstp,rtmp);
                            kk_functions_[kk1].hk_.get_value(tstp,hktmp);
                            ekintmp+=std::real((hktmp*rtmp).trace())*wt;
                        }
                }
//                ekin=ekintmp;
//                #if CDMFT_CAN_USE_MPI==1
//                 ekin=0.0;
//                 MPI::COMM_WORLD.Allreduce(&ekintmp,&ekin,1,MPI::DOUBLE,MPI_SUM);
                   MPI_Allreduce(&ekintmp,&ekin,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//                #endif
                return ekin;
}

template <class LATT>
double selfconsistency_pm<LATT>::get_Dgw_current(int tstp){
  cdmatrix rtmp(1,1),hktmp(1,1);
                double vktmp=0.0,vk=0.0;
                for(int k=0;k<lattice_model_.nk_bz_;k++){
                        double wt=lattice_model_.kweight_bz_[k];
                        int kk1=lattice_model_.idx_kk_[k];
                        if(mpi_pid_kk_[kk1]==mpi_pid_){
                                kk_functions_[kk1].Gk_r.density_matrix(tstp,rtmp);
                                lattice_model_.vk(hktmp,tstp,kk_functions_[kk1].kk_);
                                vktmp+=std::real((hktmp*rtmp).trace())*wt;
                        }
                }
//               vk=vktmp;
//               #if CDMFT_CAN_USE_MPI==1
//               vk=0.0;
//               MPI::COMM_WORLD.Allreduce(&vktmp,&vk,1,MPI::DOUBLE,MPI_SUM);
                 MPI_Allreduce(&vktmp,&vk,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//               #endif
               return vk;   
} 
/**/
// solve (1+G1c)*glatt = G  (G1c=G*Delta)
/*
void get_glatt(int nt,int tstp,cntr::herm_matrix<double> &g_latt,cntr::herm_matrix<double> &G1,
cntr::herm_matrix<double> &G1c,cntr::herm_matrix<double> &Gloc,double beta,double h,int kt){
     
  
       int nt_= nt;
       cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);
       tmpFsin.set_zero();
       tmpQsin.set_zero();
       tmpGsin.set_zero();
       cntr::vie2_timestep_sin(tstp,g_latt,tmpGsin,G1,G1c,tmpFsin,Gloc,tmpQsin,beta,h,kt);

}
**/
// first update G1c=G*Delta, G1=Delta*G, then solve (1+G1c)*glatt = G 
template <class LATT>
void selfconsistency_pm<LATT>::get_glatt(int tstp, cntr::herm_matrix<double> &Gloc){
//void get_glatt(int tstp,cntr::herm_matrix<double> &g_latt,cntr::herm_matrix<double> &Delta, cntr::function<double> &e_loc, 
//cntr::herm_matrix<double> &G1, cntr::herm_matrix<double> &G1c,cntr::herm_matrix<double> &Gloc,double beta,double h,int kt){
//        std::ostringstream name;
//        {
                int n;

//                int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);

//                 int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//                 int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);
 
//                  GREEN g_temp1(n2,ntau_,1,-1),g_temp1_cc(n2,ntau_,1,-1),
//                         g_temp2(n2,ntau_,1,-1),g_temp2_cc(n2,ntau_,1,-1);

                int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
                int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

             for(n=n1;n<=n2;n++){
                     cntr::convolution_timestep_new(n,G_Delta,Gloc,Delta_up,integration::I<double>(kt_),beta_,h_);
                     cntr::convolution_timestep_new(n,Delta_G,Delta_up,Gloc,integration::I<double>(kt_),beta_,h_);
                }   

//        GREEN g_temp1(n2,ntau_,1,-1),g_temp1_cc(n2,ntau_,1,-1);
           
//        for(n=-1;n<=n2;n++){         
 
//         g_temp1.set_timestep(n,);
//         g_temp1_cc.set_timestep(n,Gloc);
  
 
//         g_temp1.right_multiply(n,e_loc,1.0);
//         g_temp1_cc.left_multiply(n,e_loc,1.0);           
            
//         g_temp1.incr_timestep(n,G_Delta,CPLX(1.0, 0.0));
//         g_temp1_cc.incr_timestep(n,Delta_G,CPLX(1.0, 0.0));

//         }

	if(tstp==-1){
                cntr::vie2_mat_fixpoint(glatt_,G_Delta,Delta_G,Gloc,beta_,integration::I<double>(kt_),10);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(glatt_);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(glatt_,G_Delta,Delta_G,Gloc,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,glatt_,G_Delta,Delta_G,Gloc,beta_,h_,kt_);

         }



//       cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);
//       tmpFsin.set_zero();
//       tmpQsin.set_zero();
//       tmpGsin.set_zero();
  
//      cntr::vie2_timestep_sin(tstp,glatt_,tmpGsin,G_Delta,Delta_G,tmpFsin,Gloc,tmpQsin,beta_,h_,kt_);
       

//        }
//        get_glatt(nt,tstp,g_latt,G1,G1c,Gloc,beta,h,kt1);


}

template <class LATT>
void selfconsistency_pm<LATT>::optical_conductivity_bubble_DMFT(int tstp, std::vector<double> &sigma, double &sdia){

 double *chipm_local,*chipm__,sdia_local,sdia__;
 int q,n;

 chipm_local = new double [tstp+1];
 chipm__ = new double [tstp+1];
 for(n=0;n<=tstp;n++){
 chipm_local[n]=0.0;
 chipm__[n]=0.0;
 }
 sdia_local= 0.0;
 sdia__= 0.0;

 for(int k=0;k<lattice_model_.nk_bz_;k++){
    double wt=lattice_model_.kweight_bz_[k];
    int kk1=lattice_model_.idx_kk_[k];
    cdouble gret,gles,occu;
    double chipm1;
    dvector tmp(2);   
    double nk,gg,tmp1,dvkdA,vk1,vk2;
    if(mpi_pid_kk_[kk1]==mpi_pid_){
    kk_functions_[kk1].Gk_DMFT.density_matrix(tstp,occu);
    tmp(0)=kk_functions_[kk1].kk_(0);
    //tmp(1)=kk_functions_[kk1].kk_(1);
    vk1=2.0*sin(tmp(0));
    dvkdA=2.0*cos(tmp(0));
    //vk2=2.0*sin(tmp(1));
    sdia_local += (wt*occu.real()*dvkdA);
    for(n=0;n<=tstp;n++){
    kk_functions_[kk1].Gk_DMFT.get_ret(tstp,n,gret);
    kk_functions_[kk1].Gk_DMFT.get_les(n,tstp,gles);
    gg=(gret*gles).imag();
    chipm1 = -(wt*2.0*gg*vk1*vk1);
    chipm_local[n] += chipm1;
     }
  }

 }


 MPI_Allreduce(chipm_local,chipm__,tstp+1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(&sdia_local,&sdia__,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
//MPI::COMM_WORLD.Barrier();


  if(mpi_pid_==mpi_imp_){
        sigma.resize(tstp+1);
//        chipm.resize(tstp+1);
        for(n=0;n<=tstp;n++){
           double res=0.0;
           int i1,n2,kt1,kt=5;
           n2=tstp-n;
           kt1=(n2>=kt ? kt : n2);
           for(i1=0;i1<=n2;i1++){
           double gwt=integration::I<double>(kt1).gregory_weights(n2,i1);
           res += chipm__[n+i1]*gwt;
          }
          sdia=-sdia__;
//          sigma[n] = sdia__-h_*res;
          sigma[n] = -h_*res;
//          chipm[n] = chipm__[n];
      }
  
      //double res1=0,res2=0;
      //int i1,kt1,kt=5;
      //kt1=(tstp>=kt ? kt : tstp);
      // for(i1=0;i1<=tstp;i1++){
      //     double gwt=integration::I<double>(kt1).gregory_weights(tstp,i1);
      //     res1 += chipm__[i1]*gwt;
      // }

      // sdia = sdia__ - h_*res1;

   }

 }

template <class LATT>
void selfconsistency_pm<LATT>::optical_conductivity_bubble_DGW(int tstp, std::vector<double> &sigma, double &sdia){

 double *chipm_local,*chipm__,sdia_local,sdia__;
 int q,n;

 chipm_local = new double [tstp+1];
 chipm__ = new double [tstp+1];
 for(n=0;n<=tstp;n++){
 chipm_local[n]=0.0;
 chipm__[n]=0.0;
 }

 sdia_local= 0.0;
 sdia__= 0.0;


for(int k=0;k<lattice_model_.nk_bz_;k++){
    double wt=lattice_model_.kweight_bz_[k];
    int kk1=lattice_model_.idx_kk_[k];
    cdouble gret,gles,occu;
    double chipm1;
    dvector tmp(2);
    double nk,gg,tmp1,dvkdA,vk1,vk2;
    if(mpi_pid_kk_[kk1]==mpi_pid_){
     kk_functions_[kk1].Gk_r.density_matrix(tstp,occu);	    
     tmp(0)=kk_functions_[kk1].kk_(0);
     tmp(1)=kk_functions_[kk1].kk_(1);
     vk1=2.0*sin(tmp(0));
     dvkdA=2.0*cos(tmp(0));
     //vk2=2.0*sin(tmp(1));
     sdia_local += (wt*occu.real()*dvkdA);
    for(n=0;n<=tstp;n++){
      kk_functions_[kk1].Gk_r.get_ret(tstp,n,gret);
      kk_functions_[kk1].Gk_r.get_les(n,tstp,gles);
      gg=(gret*gles).imag();
      chipm1 = -(wt*2.0*gg*vk1*vk1);
      chipm_local[n] += chipm1;
     }
  
   }

 }


 MPI_Allreduce(chipm_local,chipm__,tstp+1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
 MPI_Allreduce(&sdia_local,&sdia__,1,MPI_DOUBLE_PRECISION,MPI_SUM,MPI_COMM_WORLD);
// MPI::COMM_WORLD.Barrier();


  if(mpi_pid_==mpi_imp_){
     sigma.resize(tstp+1);
    for(n=0;n<=tstp;n++){
           double res=0.0;
           int i1,n2,kt1,kt=5;
           n2=tstp-n;
           kt1=(n2>=kt ? kt : n2);
           for(i1=0;i1<=n2;i1++){
           double gwt=integration::I<double>(kt1).gregory_weights(n2,i1);
           res += chipm__[n+i1]*gwt;
          }
         //sigma[n] = sdia__-h_*res;
         sigma[n] = -h_*res;
         sdia = -sdia__	; 
      }
   }
}


} //namespace





