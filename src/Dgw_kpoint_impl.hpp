#include "Dgw_kpoint_dec.hpp"

// -----------------------------------------------------------------------
 #define CFUNC cntr::function<double>
 #define GREEN cntr::herm_matrix<double>
 #define GREEN_TSTP cntr::herm_matrix_timestep<double>
// -----------------------------------------------------------------------

namespace Dgw{

template<class LATT>
  void kpoint<LATT>::init(int nt,int ntau,int size,double beta,double h,dvector kk,LATT &latt){
    beta_ = beta;
    h_ = h;
    nt_ = nt;
    ntau_ = ntau;
    nrpa_ = size;
    hk_= CFUNC(nt_,nrpa_);
    hktilde_ = CFUNC(nt_,nrpa_);
    Vertex_c = CFUNC(nt_,nrpa_);
    Vertex_z = CFUNC(nt_,nrpa_);
    Vertex_Wk_c = CFUNC(nt_,nrpa_);
    Vertex_Wk_z = CFUNC(nt_,nrpa_);
    Vertex_ = CFUNC(nt_,nrpa_);
    Hartree_ = CFUNC(nt_,nrpa_);
    Sigma_sg_ = CFUNC(nt_,nrpa_);
//    Sigma_sd_ = CFUNC(nt_,nrpa_);
//    ret_U = CFUNC(nt_,nrpa_);
    kk_=kk;

    for(int tstp=-1;tstp<=nt_;tstp++) { 
       set_hk(tstp,latt);
       set_vertex_c(tstp,latt);
       set_vertex_z(tstp,latt);
       set_vertex_Wk_c(tstp,latt);
       set_vertex_Wk_z(tstp,latt);
       set_vertex_(tstp,latt);
       set_hktilde(tstp,latt);
    }

    Gk_DMFT=GREEN(nt_,ntau_,nrpa_,FERMION);
    Gk_tilde=GREEN(nt_,ntau_,nrpa_,FERMION);
    Gk_dual=GREEN(nt_,ntau_,nrpa_,FERMION);
    Sigma_dual=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_3temp=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_3temp_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
    Sigma_dual_mix=GREEN(-1,ntau_,nrpa_,FERMION);  

    Wk_bare_c=GREEN(nt_,ntau_,nrpa_,BOSON);
    Wk_bare_z=GREEN(nt_,ntau_,nrpa_,BOSON);
    Wk_dual_c=GREEN(nt_,ntau_,nrpa_,BOSON);
    Wk_dual_z=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_dual=GREEN(nt_,ntau_,nrpa_,BOSON);
    Convo_1temp=GREEN(nt_,ntau_,nrpa_,BOSON);
    Convo_2temp=GREEN(nt_,ntau_,nrpa_,BOSON);
    Convo_1temp_cc=GREEN(nt_,ntau_,nrpa_,BOSON);
    Convo_2temp_cc=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_mix=GREEN(-1,ntau_,nrpa_,BOSON);
    
  }
template<class LATT>
  void kpoint<LATT>::set_hk(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.hk(hktmp,tstp,kk_);
    hk_.set_value(tstp,hktmp);
  }

template<class LATT>
  void kpoint<LATT>::set_vertex_c(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.V_qc(hktmp,tstp,kk_);
    Vertex_c.set_value(tstp,hktmp);
  }

template<class LATT>
  void kpoint<LATT>::set_vertex_z(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.V_qz(hktmp,tstp,kk_);
    Vertex_z.set_value(tstp,hktmp);
  }


template<class LATT>
  void kpoint<LATT>::set_vertex_Wk_c(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.Wk_qc(hktmp,tstp,kk_);
    Vertex_Wk_c.set_value(tstp,hktmp);
  }

template<class LATT>
  void kpoint<LATT>::set_vertex_Wk_z(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.Wk_qz(hktmp,tstp,kk_);
    Vertex_Wk_z.set_value(tstp,hktmp);
  }


template<class LATT>
  void kpoint<LATT>::set_vertex_(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
    cdmatrix hktmp(nrpa_,nrpa_);
    latt.V_q(hktmp,tstp,kk_);
    Vertex_.set_value(tstp,hktmp);
  }

template<class LATT>
  void kpoint<LATT>::set_hktilde(int tstp, LATT &latt){
    assert(-1<=tstp && tstp<=nt_);
//    cdmatrix hktmp(nrpa_,nrpa_);
//    latt.V_q(hktmp,tstp,kk_);
    hktilde_.set_zero();
}



template<class LATT>
void kpoint<LATT>::Get_G_latt_DMFT(int tstp, int kt_, cntr::herm_matrix<double> &g_latt){

    int n;

//     int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);

//     int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//     int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);



    int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

//     int n2=(tstp<=kt_ && tstp>=0 ? kt_ : tstp); 

    GREEN g_eps(n2,ntau_,1,-1),eps_g(n2,ntau_,1,-1);
   
//    CFUNC h_tilde(n2,1);

//     GREEN g_eps(nt_,ntau_,1,-1),eps_g(nt_,ntau_,1,-1);
  

    for(n=-1;n<=n2;n++){
      g_eps.set_timestep(n,g_latt);
      eps_g.set_timestep(n,g_latt);
      
       g_eps.right_multiply(n,hktilde_,-1.0);
       eps_g.left_multiply(n,hktilde_,-1.0);

//      g_eps.right_multiply(n,hk_,-1.0);
//      eps_g.left_multiply(n,hk_,-1.0);
    }

  
        if(tstp==-1){
                cntr::vie2_mat_fixpoint(Gk_DMFT,g_eps,eps_g,g_latt,beta_,integration::I<double>(kt_),10);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Gk_DMFT);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Gk_DMFT,g_eps,eps_g,g_latt,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Gk_DMFT,g_eps,eps_g,g_latt,beta_,h_,kt_);

         }


//    cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);
//    tmpFsin.set_zero();
//    tmpQsin.set_zero();
//    tmpGsin.set_zero();


//   cntr::vie2_timestep_sin(tstp,Gk_DMFT,tmpGsin,g_eps,eps_g,tmpFsin,g_latt,tmpQsin,beta_,h_,kt_);
        
}
/**/
template<class LATT>
void kpoint<LATT>::step_Wk_dual(int tstp, int kt_){

        int n;
        int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);

//        int n1=(tstp<=kt_ && tstp>=0 ? 0 : tstp);
//        int n2=(tstp<=kt_ && tstp>=0 ? kt_ : tstp);

        int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
        int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);


        for(n=n1;n<=n2;n++){
             cntr::convolution_timestep_new(n,Convo_1temp,Wk_bare_c,Pi_dual,integration::I<double>(kt1),beta_,h_); //kt1
             cntr::convolution_timestep_new(n,Convo_1temp_cc,Pi_dual,Wk_bare_c,integration::I<double>(kt1),beta_,h_);
             cntr::convolution_timestep_new(n,Convo_2temp,Wk_bare_z,Pi_dual,integration::I<double>(kt1),beta_,h_);
             cntr::convolution_timestep_new(n,Convo_2temp_cc,Pi_dual,Wk_bare_z,integration::I<double>(kt1),beta_,h_);
         }

        

//        GREEN g_temp1(nt_,ntau_,1,1),g_temp1_cc(nt_,ntau_,1,1),
//                 g_temp2(nt_,ntau_,1,1),g_temp2_cc(nt_,ntau_,1,1);


        GREEN g_temp1(n2,ntau_,1,1),g_temp1_cc(n2,ntau_,1,1),
                 g_temp2(n2,ntau_,1,1),g_temp2_cc(n2,ntau_,1,1),
                  g_temp3(n2,ntau_,1,1),g_temp4(n2,ntau_,1,1);

        for(n=-1;n<=n2;n++){

         g_temp1.set_timestep(n,Pi_dual);
         g_temp1_cc.set_timestep(n,Pi_dual);

         g_temp2.set_timestep(n,Pi_dual);
         g_temp2_cc.set_timestep(n,Pi_dual);
     
         g_temp1.left_multiply(n,Vertex_Wk_c,-1.0);
         g_temp1_cc.right_multiply(n,Vertex_Wk_c,-1.0);

         g_temp2.left_multiply(n,Vertex_Wk_z,-1.0);
         g_temp2_cc.right_multiply(n,Vertex_Wk_z,-1.0);
          

         g_temp1.incr_timestep(n,Convo_1temp,CPLX(-1.0, 0.0));
         g_temp1_cc.incr_timestep(n,Convo_1temp_cc,CPLX(-1.0, 0.0));

         g_temp2.incr_timestep(n,Convo_2temp,CPLX(-1.0, 0.0));
         g_temp2_cc.incr_timestep(n,Convo_2temp_cc,CPLX(-1.0, 0.0));

//         g_temp1.set_timestep_zero(n);
//         g_temp1_cc.set_timestep_zero(n);

//         g_temp2.set_timestep_zero(n);
//         g_temp2_cc.set_timestep_zero(n);

         g_temp3.set_timestep(n,g_temp1);
	 g_temp3.right_multiply(n,Vertex_Wk_c,-1.0);

	 g_temp4.set_timestep(n,g_temp2);  
	 g_temp4.right_multiply(n,Vertex_Wk_z,-1.0);

	 g_temp3.incr_timestep(n,Wk_bare_c,CPLX(1.0, 0.0));
	 g_temp4.incr_timestep(n,Wk_bare_z,CPLX(1.0, 0.0));

//         g_temp2_cc.set_timestep_zero(n);

         
//           Wk_dual_z.set_timestep_zero(n);  // off z chanel 
//           Wk_dual_c.set_timestep_zero(n); // off c channel

//	  if(n>=0){

//          cdouble m_r1,m_rc1,m_c1,m_cc1;
//	  cdouble m_r2,m_rc2,m_c2,m_cc2;
//	  cdouble me1, me2, me3, me4;

//          g_temp1.get_ret(n,n,m_r1);
//          g_temp1_cc.get_ret(n,n,m_c1);
//	  g_temp2.get_ret(n,n,m_r2);
//          g_temp2_cc.get_ret(n,n,m_c2);

//          m_rc1=std::conj(m_r1);
//          m_cc1=std::conj(m_c1);
//	  m_rc2=std::conj(m_r2);
//          m_cc2=std::conj(m_c2);

//          me1 = (m_r1 + m_rc1)*0.50;
//          me2 = (m_c1 + m_cc1)*0.50;
//          me3 = (m_r2 + m_rc2)*0.50;
//          me4 = (m_c2 + m_cc2)*0.50;


//          g_temp1.set_ret(n,n,me1);
//          g_temp1_cc.set_ret(n,n,me2);
//          g_temp2.set_ret(n,n,me3);
//          g_temp2_cc.set_ret(n,n,me4);

//          }


	}

  
	  if(tstp==-1){
                cntr::vie2_mat_fixpoint(Wk_dual_c,g_temp1,g_temp1_cc,g_temp3,beta_,integration::I<double>(kt_),10);
                cntr::vie2_mat_fixpoint(Wk_dual_z,g_temp2,g_temp2_cc,g_temp4,beta_,integration::I<double>(kt_),10);
//         }
//         else if(tstp==0){
//                cntr::set_t0_from_mat(Wk_dual_c);
//                cntr::set_t0_from_mat(Wk_dual_z);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Wk_dual_c,g_temp1,g_temp1_cc,g_temp3,integration::I<double>(kt1),beta_,h_);
               cntr::vie2_start(Wk_dual_z,g_temp2,g_temp2_cc,g_temp4,integration::I<double>(kt1),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Wk_dual_c,g_temp1,g_temp1_cc,g_temp3,beta_,h_,kt_);
             cntr::vie2_timestep(tstp,Wk_dual_z,g_temp2,g_temp2_cc,g_temp4,beta_,h_,kt_);

         }
   

//        cntr::function<double> tmpF1sin(nt_,1), tmpF2sin(nt_,1), tmpQ1sin(nt_,1), 
//                 tmpQ2sin(nt_,1), tmpG1sin(nt_,1), tmpG2sin(nt_,1);

//         tmpF1sin.set_zero();
//         tmpQ1sin.set_zero();
//         tmpQ2sin.set_zero();

//         for(n=-1;n<=n2;n++){
//         cdmatrix At1(1,1),At2(1,1);
//         Vertex_Wk_c.get_value(n,At1);
//         Vertex_Wk_z.get_value(n,At2);    
//         tmpQ1sin.set_value(n,At1);
//         tmpQ2sin.set_value(n,At2);   
//         tmpQ1sin.set_matrixelement(0,0,Vertex_Wk_c,0,0);
//         }
//         tmpG1sin.set_zero();
  
//        cntr::vie2_timestep_sin(tstp,Wk_dual_c,tmpG1sin,g_temp1,g_temp1_cc,tmpF1sin,Wk_bare_c,tmpQ1sin,beta_,h_,kt1);

//        tmpF2sin.set_zero();
//        tmpQ2sin.set_matrixelement(0,0,Vertex_Wk_z,0,0);
//        tmpG2sin.set_zero();

        
//        cntr::vie2_timestep_sin(tstp,Wk_dual_z,tmpG2sin,g_temp2,g_temp2_cc,tmpF2sin,Wk_bare_z,tmpQ2sin,beta_,h_,kt1);

        
/**/
}

template<class LATT>
void kpoint<LATT>::step_Gk_dual(int tstp, int kt_){
  
         int n;
         int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);
         
//         int n1=(tstp<=kt_ && tstp>=0 ? 0 : tstp);
//         int n2=(tstp<=kt_ && tstp>=0 ? kt_ : tstp);

        int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
        int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);
     
         for(n=n1;n<=n2;n++){
             cntr::convolution_timestep_new(n,Convo_3temp,Gk_tilde,Sigma_dual,integration::I<double>(kt1),beta_,h_); //kt1
             cntr::convolution_timestep_new(n,Convo_3temp_cc,Sigma_dual,Gk_tilde,integration::I<double>(kt1),beta_,h_);
         }


         GREEN g_temp1(n2,ntau_,1,-1),g_temp1_cc(n2,ntau_,1,-1);

         for(n=-1;n<=n2;n++){

          g_temp1.set_timestep(n,Gk_tilde);
          g_temp1_cc.set_timestep(n,Gk_tilde);

          g_temp1.right_multiply(n,Sigma_sg_,-1.0);
          g_temp1_cc.left_multiply(n,Sigma_sg_,-1.0);

          g_temp1.incr_timestep(n,Convo_3temp,CPLX(-1.0, 0.0));
          g_temp1_cc.incr_timestep(n,Convo_3temp_cc,CPLX(-1.0, 0.0));

//	  if(n>=0){
//          cdouble m_r,m_rc,m_c,m_cc;
//	  cdouble me1,me2;
//	  g_temp1.get_ret(n,n,m_r);
//	  g_temp1_cc.get_ret(n,n,m_c);
//	  m_rc=std::conj(m_r);
//	  m_cc=std::conj(m_c);
//          me1 = (m_r + m_rc)*0.50;
//	  me2 = (m_c + m_cc)*0.50;
//	  g_temp1.set_ret(n,n,me1);
//          g_temp1_cc.set_ret(n,n,me2);
//          }		  

//	  g_temp1.set_timestep_zero(n);
//	  g_temp1_cc.set_timestep_zero(n);
         }

//        cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);

//         tmpFsin.set_zero();
//         tmpQsin.set_zero();
//         tmpGsin.set_zero();

	 if(tstp==-1){

		cntr::vie2_mat_fixpoint(Gk_dual,g_temp1,g_temp1_cc,Gk_tilde,beta_,integration::I<double>(kt_),10); 
//	 }
//	 else if(tstp==0){
//		cntr::set_t0_from_mat(Gk_dual);
	}
	 else if(tstp<=kt_){
               cntr::vie2_start(Gk_dual,g_temp1,g_temp1_cc,Gk_tilde,integration::I<double>(kt1),beta_,h_);
	 }
	 else{

	     cntr::vie2_timestep(tstp,Gk_dual,g_temp1,g_temp1_cc,Gk_tilde,beta_,h_,kt_);

	 }


//***       cntr::vie2_timestep_sin(tstp,Gk_dual,tmpGsin,g_temp1,g_temp1_cc,tmpFsin,Gk_tilde,tmpQsin,beta_,h_,kt1);

}
/**/
template<class LATT>
void kpoint<LATT>::step_Gk_dual_with_error(int tstp, int kt_, double &err3){
        
                GREEN_TSTP gtmp(tstp,ntau_,1,-1);
                Gk_dual.get_timestep(tstp,gtmp);
                if(tstp != 0){
                step_Gk_dual(tstp,kt_);
		}
		else{
                cntr::set_t0_from_mat(Gk_dual);
		}			
                err3=0.0;
                err3=cntr::distance_norm2(tstp,gtmp,Gk_dual);
 }
/**/
template<class LATT>
void kpoint<LATT>::step_Wk_dual_with_error(int tstp, int kt_, double &err1,double &err2){

                GREEN_TSTP gtmp1(tstp,ntau_,1,1), gtmp2(tstp,ntau_,1,1);
                Wk_dual_c.get_timestep(tstp,gtmp1);
                Wk_dual_z.get_timestep(tstp,gtmp2);
		if(tstp != 0){
                step_Wk_dual(tstp,kt_);
		}
		else{
		cntr::set_t0_from_mat(Wk_dual_c);
                cntr::set_t0_from_mat(Wk_dual_z);
                }
                err1=0.0;
                err2=0.0;
                err1=cntr::distance_norm2(tstp,gtmp1,Wk_dual_c);
                err2=cntr::distance_norm2(tstp,gtmp2,Wk_dual_z);
               
                  
}
template<class LATT>
void kpoint<LATT>::init_observables(){

    Gk_r=GREEN(nt_,ntau_,nrpa_,FERMION);
    Sigma_G=GREEN(nt_,ntau_,nrpa_,FERMION);
    Sigma_G_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
    G_Sigma_G=GREEN(nt_,ntau_,nrpa_,FERMION);
    G_Sigma_sg_G=GREEN(nt_,ntau_,nrpa_,FERMION);
    TK_bar=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_3temp=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_3temp_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_4temp=GREEN(nt_,ntau_,nrpa_,FERMION);
    Convo_4temp_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
    sigma_bar = GREEN(nt_,ntau_,nrpa_,FERMION);

//    Sigma_D=GREEN(nt_,ntau_,nrpa_,FERMION);
//    Sigma_D_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
//    G_Sigma_G_D=GREEN(nt_,ntau_,nrpa_,FERMION);
//    G_Sigma_G_D_cc=GREEN(nt_,ntau_,nrpa_,FERMION);
//    G_Sigma_sd_G=GREEN(nt_,ntau_,nrpa_,FERMION);
//    G_Si_G_D=GREEN(nt_,ntau_,nrpa_,FERMION);
//    G_Si_G_D_cc=GREEN(nt_,ntau_,nrpa_,FERMION);     

 
       
    Chi_rc=GREEN(nt_,ntau_,nrpa_,BOSON);
    Chi_rz=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_bar_c=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_bar_z=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_latt_c=GREEN(nt_,ntau_,nrpa_,BOSON);
    Pi_latt_z=GREEN(nt_,ntau_,nrpa_,BOSON);

}

template<class LATT>
void kpoint<LATT>::Get_real_Glatt(int tstp, int kt_,cntr::herm_matrix<double> &g_loc_, 
                       cntr::herm_matrix<double> &Delta_){

     int n;

//     int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);

//     int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//     int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);

     int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
     int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

     for(n=n1;n<=n2;n++){    
      cntr::convolution_timestep_new(n,Sigma_G,g_loc_,Sigma_dual,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Sigma_G_cc,Sigma_dual,g_loc_,integration::I<double>(kt_),beta_,h_);
     }

      for(n=n1;n<=n2;n++){
        cntr::convolution_timestep_new(n,G_Sigma_G,Sigma_G,Sigma_G_cc,g_loc_,g_loc_,integration::I<double>(kt_),beta_,h_);
      }

      for(n=n1;n<=n2;n++){
      cntr::convolution_timestep_new(n,G_Sigma_sg_G,g_loc_,Sigma_sg_,g_loc_,integration::I<double>(kt_),beta_,h_);
//      G_Sigma_sg_G.incr_timestep(n,G_Sigma_G,CPLX(1.0,0.0));
      }

      for(n=n1;n<=n2;n++){
      TK_bar.set_timestep(n,G_Sigma_sg_G);
      TK_bar.incr_timestep(n,g_loc_,CPLX(1.0,0.0));
      TK_bar.incr_timestep(n,G_Sigma_G,CPLX(1.0,0.0));
//      TK_bar.set_timestep(n,g_loc_);
      }

      for(n=n1;n<=n2;n++){     
      cntr::convolution_timestep_new(n,Convo_3temp,TK_bar,Delta_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Convo_3temp_cc,Delta_,TK_bar,integration::I<double>(kt_),beta_,h_);
     }

   GREEN g_temp(n2,ntau_,1,-1),g_temp_cc(n2,ntau_,1,-1);
  
    
   for(n=-1;n<=n2;n++){   
    
    g_temp.set_timestep(n,TK_bar);
    g_temp_cc.set_timestep(n,TK_bar);

    g_temp.right_multiply(n,hktilde_,-1.0);
    g_temp_cc.left_multiply(n,hktilde_,-1.0);

//    g_temp.right_multiply(n,hk_,-1.0);
//    g_temp_cc.left_multiply(n,hk_,-1.0);

    g_temp.incr_timestep(n,Convo_3temp,CPLX(1.0, 0.0));
    g_temp_cc.incr_timestep(n,Convo_3temp_cc,CPLX(1.0, 0.0));
   
    }

        if(tstp==-1){
           cntr::vie2_mat_fixpoint(Gk_r,g_temp,g_temp_cc,TK_bar,beta_,integration::I<double>(kt_),10);
	   cntr::force_matsubara_hermitian(Gk_r);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Gk_r);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Gk_r,g_temp,g_temp_cc,TK_bar,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Gk_r,g_temp,g_temp_cc,TK_bar,beta_,h_,kt_);

         }

   
//   cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);

//         tmpFsin.set_zero();
//         tmpQsin.set_zero();
//         tmpGsin.set_zero();

//     cntr::vie2_timestep_sin(tstp,Gk_r,tmpGsin,g_temp,g_temp_cc,tmpFsin,TK_bar,tmpQsin,beta_,h_,kt_);
 

}
/*
template<class LATT>
void kpoint<LATT>::Get_real_Glatt(int tstp, int kt_,cntr::herm_matrix<double> &g_loc_,
                       cntr::herm_matrix<double> &Delta_){
     int n;

     int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
     int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);

     for(n=n1;n<=n2;n++){
      cntr::convolution_timestep_new(n,Sigma_G,g_loc_,Sigma_dual,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Sigma_G_cc,Sigma_dual,g_loc_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Sigma_D,g_loc_,Delta_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Sigma_D_cc,Delta_,g_loc_,integration::I<double>(kt_),beta_,h_);
     }

      for(n=n1;n<=n2;n++){
        cntr::convolution_timestep_new(n,G_Sigma_G,Sigma_G,Sigma_G_cc,g_loc_,g_loc_,integration::I<double>(kt_),beta_,h_);
        cntr::convolution_timestep_new(n,G_Sigma_G_D,Sigma_G,Sigma_G_cc,Sigma_D,Sigma_D_cc,integration::I<double>(kt_),beta_,h_);
        cntr::convolution_timestep_new(n,G_Sigma_G_D_cc,Sigma_D_cc,Sigma_D,Sigma_G_cc,Sigma_G,integration::I<double>(kt_),beta_,h_);
      }

      for(n=n1;n<=n2;n++){
      cntr::convolution_timestep_new(n,G_Sigma_sg_G,g_loc_,Sigma_sg_,g_loc_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,G_Sigma_sd_G,g_loc_,Sigma_sd_,g_loc_,integration::I<double>(kt_),beta_,h_);
      }

      for(n=n1;n<=n2;n++){
      cntr::convolution_timestep_new(n,G_Si_G_D,G_Sigma_sg_G,G_Sigma_sd_G,Delta_,Delta_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,G_Si_G_D_cc,Delta_,Delta_,G_Sigma_sg_G,G_Sigma_sd_G,integration::I<double>(kt_),beta_,h_);
      }

      
      for(n=n1;n<=n2;n++){
      TK_bar.set_timestep(n,G_Sigma_G);
      TK_bar.incr_timestep(n,G_Sigma_sg_G,CPLX(1.0,0.0));
      TK_bar.incr_timestep(n,g_loc_,CPLX(1.0,0.0));
      }
   
    GREEN g_temp(n2,ntau_,1,-1),g_temp_cc(n2,ntau_,1,-1);

    GREEN g_temp1(n2,ntau_,1,-1),g_temp1_cc(n2,ntau_,1,-1);   


    for(n=-1;n<=n2;n++){

    g_temp.set_timestep(n,g_loc_);
    g_temp_cc.set_timestep(n,g_loc_);

    g_temp1.set_timestep(n,G_Sigma_G);
    g_temp1_cc.set_timestep(n,G_Sigma_G);

    g_temp1.incr_timestep(n,G_Sigma_sg_G,CPLX(1.0, 0.0));
    g_temp1_cc.incr_timestep(n,G_Sigma_sg_G,CPLX(1.0, 0.0));

    g_temp.right_multiply(n,hk_,-1.0);
    g_temp_cc.left_multiply(n,hk_,-1.0);

    g_temp1.right_multiply(n,hk_,-1.0);
    g_temp1_cc.left_multiply(n,hk_,-1.0);
    
    g_temp.incr_timestep(n,Sigma_D,CPLX(1.0, 0.0));
  
    g_temp.incr_timestep(n,G_Sigma_G_D,CPLX(1.0, 0.0));

    g_temp.incr_timestep(n,G_Si_G_D,CPLX(1.0, 0.0));

    g_temp.incr_timestep(n,g_temp1,CPLX(1.0, 0.0));

    g_temp_cc.incr_timestep(n,Sigma_D_cc,CPLX(1.0, 0.0));
   
    g_temp_cc.incr_timestep(n,G_Sigma_G_D_cc,CPLX(1.0, 0.0));

    g_temp_cc.incr_timestep(n,G_Si_G_D_cc,CPLX(1.0, 0.0));

    g_temp_cc.incr_timestep(n,g_temp1_cc,CPLX(1.0, 0.0));

    }

    cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);

         tmpFsin.set_zero();
         tmpQsin.set_zero();
         tmpGsin.set_zero();

     cntr::vie2_timestep_sin(tstp,Gk_r,tmpGsin,g_temp,g_temp_cc,tmpFsin,TK_bar,tmpQsin,beta_,h_,kt_);

}
*/
template<class LATT>
void kpoint<LATT>::Get_real_chi_latt(int tstp, int kt_, 
       cntr::herm_matrix<double> &pi_c, cntr::herm_matrix<double> &pi_z, cntr::function<double> &U_){
     
         int n;
 
//         int kt1 = (tstp==-1 || tstp>=kt_ ? kt_ : tstp);

//         int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//         int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);


         int n2 = (tstp==-1 || tstp>kt_ ? tstp : kt_);
 
//       int n2=(tstp<=kt_ && tstp>=0 ? kt_ : tstp);      

       GREEN g_bose1(n2,ntau_,1,1),g_bose2(n2,ntau_,1,1),
        g_bose1_cc(n2,ntau_,1,1),g_bose2_cc(n2,ntau_,1,1);    
  
       for(n=-1;n<=n2;n++){

        g_bose1.set_timestep(n,Pi_dual);
        g_bose1_cc.set_timestep(n,Pi_dual);

        g_bose2.set_timestep(n,Pi_dual);
        g_bose2_cc.set_timestep(n,Pi_dual);

        g_bose1.right_multiply(n,U_,0.25);
        g_bose1_cc.left_multiply(n,U_,0.25);

        g_bose2.right_multiply(n,U_,-0.25);
        g_bose2_cc.left_multiply(n,U_,-0.25);
        }


        if(tstp==-1){
           cntr::vie2_mat_fixpoint(Pi_bar_c,g_bose1,g_bose1_cc,Pi_dual,beta_,integration::I<double>(kt_),10);
           cntr::vie2_mat_fixpoint(Pi_bar_z,g_bose2,g_bose2_cc,Pi_dual,beta_,integration::I<double>(kt_),10);
	   cntr::force_matsubara_hermitian(Pi_bar_c);
           cntr::force_matsubara_hermitian(Pi_bar_z);

         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Pi_bar_c);
                cntr::set_t0_from_mat(Pi_bar_z);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Pi_bar_c,g_bose1,g_bose1_cc,Pi_dual,integration::I<double>(kt_),beta_,h_);
               cntr::vie2_start(Pi_bar_z,g_bose2,g_bose2_cc,Pi_dual,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Pi_bar_c,g_bose1,g_bose1_cc,Pi_dual,beta_,h_,kt_);
             cntr::vie2_timestep(tstp,Pi_bar_z,g_bose2,g_bose2_cc,Pi_dual,beta_,h_,kt_);

         }


//       cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);

//        tmpFsin.set_zero();
//        tmpQsin.set_zero();
//        tmpGsin.set_zero();


//        cntr::vie2_timestep_sin(tstp,Pi_bar_c,tmpGsin,g_bose1,g_bose1_cc,tmpFsin,
//                            Pi_dual,tmpQsin,beta_,h_,kt_);

//        tmpFsin.set_zero();
//        tmpQsin.set_zero();
//        tmpGsin.set_zero();

//        cntr::vie2_timestep_sin(tstp,Pi_bar_z,tmpGsin,g_bose2,g_bose2_cc,tmpFsin,
//                      Pi_dual,tmpQsin,beta_,h_,kt_);

        for(n=-1;n<=n2;n++){

        Pi_latt_c.set_timestep(n,Pi_bar_c);
        Pi_latt_z.set_timestep(n,Pi_bar_z);
        
        Pi_latt_c.incr_timestep(n,pi_c,CPLX(1.0, 0.0));
        Pi_latt_z.incr_timestep(n,pi_z,CPLX(1.0, 0.0));

        g_bose1.set_timestep(n,Pi_latt_c);
        g_bose1_cc.set_timestep(n,Pi_latt_c);

        g_bose2.set_timestep(n,Pi_latt_z);
        g_bose2_cc.set_timestep(n,Pi_latt_z);

        g_bose1.right_multiply(n,Vertex_c,-1.0);
        g_bose1_cc.left_multiply(n,Vertex_c,-1.0);

        g_bose2.right_multiply(n,Vertex_z,-1.0);
        g_bose2_cc.left_multiply(n,Vertex_z,-1.0);


        }

        if(tstp==-1){
           cntr::vie2_mat_fixpoint(Chi_rc,g_bose1,g_bose1_cc,Pi_latt_c,beta_,integration::I<double>(kt_),10);
	   cntr::vie2_mat_fixpoint(Chi_rz,g_bose2,g_bose2_cc,Pi_latt_z,beta_,integration::I<double>(kt_),10);
	   cntr::force_matsubara_hermitian(Chi_rc);
	   cntr::force_matsubara_hermitian(Chi_rz);
         }
         else if(tstp==0){
                cntr::set_t0_from_mat(Chi_rc);
		cntr::set_t0_from_mat(Chi_rz);
        }
         else if(tstp<=kt_){
               cntr::vie2_start(Chi_rc,g_bose1,g_bose1_cc,Pi_latt_c,integration::I<double>(kt_),beta_,h_);
	       cntr::vie2_start(Chi_rz,g_bose2,g_bose2_cc,Pi_latt_z,integration::I<double>(kt_),beta_,h_);
         }
         else{

             cntr::vie2_timestep(tstp,Chi_rc,g_bose1,g_bose1_cc,Pi_latt_c,beta_,h_,kt_);
	     cntr::vie2_timestep(tstp,Chi_rz,g_bose2,g_bose2_cc,Pi_latt_z,beta_,h_,kt_);

         }

//        tmpFsin.set_zero();
//        tmpQsin.set_zero();
//        tmpGsin.set_zero();

//        cntr::vie2_timestep_sin(tstp,Chi_rc,tmpGsin,g_bose1,g_bose1_cc,tmpFsin,
//                       Pi_latt_c,tmpQsin,beta_,h_,kt_);

//        tmpFsin.set_zero();
//        tmpQsin.set_zero();
//        tmpGsin.set_zero();


//       cntr::vie2_timestep_sin(tstp,Chi_rz,tmpGsin,g_bose2,g_bose2_cc,tmpFsin,
//                       Pi_latt_z,tmpQsin,beta_,h_,kt_);

    }

template<class LATT>
void kpoint<LATT>::Get_sigma_correction(int tstp, int kt_, cntr::herm_matrix<double> &g_loc_){

  int n;

//  int kt1 = (tstp==-1 || tstp>kt_ ? kt_ : tstp);

//  int n1=(tstp<=kt1 && tstp>=0 ? 0 : tstp);
//  int n2=(tstp<=kt1 && tstp>=0 ? kt1 : tstp);

  int n1=(tstp==-1 || tstp>kt_ ? tstp : 0);
  int n2=(tstp==-1 || tstp>kt_ ? tstp : kt_);
 
  for(n=n1;n<=n2;n++){
      cntr::convolution_timestep_new(n,Convo_4temp,Sigma_dual,g_loc_,integration::I<double>(kt_),beta_,h_);
      cntr::convolution_timestep_new(n,Convo_4temp_cc,g_loc_,Sigma_dual,integration::I<double>(kt_),beta_,h_);
     }

   
  GREEN g_temp(n2,ntau_,1,-1),g_temp_cc(n2,ntau_,1,-1);

    for(n=-1;n<=n2;n++){

        g_temp.set_timestep(n,g_loc_);
        g_temp_cc.set_timestep(n,g_loc_);

        g_temp.left_multiply(n,Sigma_sg_,1.0);
        g_temp_cc.right_multiply(n,Sigma_sg_,1.0);

        g_temp.incr_timestep(n,Convo_4temp,CPLX(1.0, 0.0));
        g_temp_cc.incr_timestep(n,Convo_4temp_cc,CPLX(1.0, 0.0));
       
       }

     cntr::function<double> tmpFsin(nt_,1), tmpQsin(nt_,1), tmpGsin(nt_,1);

        tmpFsin.set_zero();
        tmpQsin.set_zero();

         for(n=-1;n<=n2;n++){
         cdmatrix At1(1,1);
         Sigma_sg_.get_value(n,At1);
         tmpQsin.set_value(n,At1);
        }
//        tmpQsin.set_matrixelement(0,0,Sigma_sg_,0,0);
        tmpGsin.set_zero();


     cntr::vie2_timestep_sin(tstp,sigma_bar,tmpGsin,g_temp,g_temp_cc,tmpFsin,
                            Sigma_dual,tmpQsin,beta_,h_,kt_);

    }   


}
