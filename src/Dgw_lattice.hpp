#ifndef _FLEX_LATTICES_H
#define _FLEX_LATTICES_H

#pragma once

#include <cmath>
#include <cassert>
#include <iostream>
#include <complex>
#include <vector>

#include <cntr/cntr.hpp>
#include <cntr/hdf5/hdf5_interface.hpp>
#include <cntr/hdf5/hdf5_interface_cntr.hpp>


#define ASSERT_0 1
#define ASSERT_1 1


namespace Dgw {

class lattice_2d{
/*///////////////////////////////////////////////////////////////////////////

The lattice should store the setup of kpoints:

* There is a "big" mesh of kpoints 0 ... nk_bz_-1 in the Brilluoin zone

* The BZ is given by the equidistant mesh
  kk = kk0 + i1*kk1/L1 + i2*kk2/L2, i1,  i1 \in {0,...,L1-1},  i2 \in {0,...,L2-1}
  which are labelled by
 
* each point k in the BZ has one "symmetry representative" R(k), so that
  all properties depend only on R(k), e.g., G_{k}=G_{R(k)}, etc.
  (later, Green functions are only stored for representatives)

* kpoints_[0...nk_-1] is the list of all representatives

* idx_kk_[0...nk_bz_-1] is the index of a kpoint in the representative list

There is a function to add kpoints and project back to BZ

In addition the lattice stores information about the dispersion functions
at a given time mesh

///////////////////////////////////////////////////////////////////////////*/
public:
    int nt_;     // number of timepoints
    int L_;
    int nk_,nk_bz_;  // number of mesh-points in full BZ
    std::vector<dvector>  kpoints_bz_;  // 0...nk_bz_-1, all kpoints
    std::vector<double>   kweight_bz_;  // 0...nk_bz_-1, normalized to 1
    std::vector<dvector>  kpoints_;     // 0...nk_-1, representatives
    std::vector<int>  idx_kk_;          // 0...nk_-1, see above
    std::vector<int>  idx_inv_;          // get =a q index for a reptresentative
    std::vector<double>  A_,E_;  // A_[t+1] = vector potential at time t= -1,...,nt, symmatry of field set by field_symmetry_
    int field_symmetry_;  // =0 for n field, 1 for (1,0)-polarized,  2 for (1,1) polarized
    /////////////////////////////////////////////
    // simple but not efficient (no FFT) way to get real-space quantities:
    int nR_;
    std::vector<dvector>  Rpoints_;
//    typedef cntr::function<double> cfunc1x1;
//    cfunc1x1 U_;
    //////////////////////////////////////////////////////////////////////
	// MODEL PARAMETERS
    double mu_;
    std::vector<double>   tx_, V_, U0_; // time-dependent hopping (preserves symmetry, though ...)
    std::vector<double>   ty_; // time-dependent hopping (preserves symmetry, though ...)
    int idx_ix(int ik){ return ik/L_;}
    int idx_iy(int ik){ return ik%L_;}
    int idx(int ix,int iy){ return ix*L_+iy;}
    void init(int L, int nt, int field_symmetry=0){
        double dk;
        nt_=nt;
        dvector dkx(2),dky(2);
        field_symmetry_=field_symmetry;
		L_=L;
        nk_bz_=L_*L_;
//        dk=2.0*PI/L_;
        dk=(2.0*CNTR_PI)/((double) L_);
//        std::cout << std::setprecision(10) << CNTR_PI <<"\t"<< dk << std::endl;
		// set vectors in full BZ:
	kweight_bz_.resize(nk_bz_);
	kpoints_bz_.resize(nk_bz_);
        kpoints_.resize(nk_bz_);
        idx_kk_.resize(nk_bz_);
        idx_inv_.resize(nk_bz_);
        if(field_symmetry_==0){
            nk_=0;
            for(int ik=0;ik<nk_bz_;ik++){
                dvector tmp(2);
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                tmp(0)=(ix*1.0)*dk;
                tmp(1)=(iy*1.0)*dk;
                kpoints_bz_[ik]=tmp;
                kweight_bz_[ik]=1.0/(double(nk_bz_));
                if(ix<=L_/2 && iy<=L_/2 && ix>=iy){
                    kpoints_[nk_]=tmp;
                    idx_kk_[ik]=nk_;
                    idx_inv_[nk_]=ik;
                    nk_++;
                }
            }
            for(int ik=0;ik<nk_bz_;ik++){
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                // map the point:
                if(ix>L_/2) ix=L_-ix;
                if(iy>L_/2) iy=L_-iy;
                if(ix<iy){
                    int itmp=ix;
                    ix=iy;
                    iy=itmp;
                }
                int ik1=idx(ix,iy);
                idx_kk_[ik]=idx_kk_[ik1];
            }
        }else if(field_symmetry_==1){
            nk_=0;
            for(int ik=0;ik<nk_bz_;ik++){
                dvector tmp(2);
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                tmp(0)=(ix*1.0)*dk;
                tmp(1)=(iy*1.0)*dk;
                kpoints_bz_[ik]=tmp;
                kweight_bz_[ik]=1.0/nk_bz_;
                if(iy<=L_/2){
                    kpoints_[nk_]=tmp;
                    idx_kk_[ik]=nk_;
                    idx_inv_[nk_]=ik;
                    nk_++;
                }
            }
            for(int ik=0;ik<nk_bz_;ik++){
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                // map the point:
                if(iy>L_/2) iy=L_-iy;
                int ik1=idx(ix,iy);
                idx_kk_[ik]=idx_kk_[ik1];
            }
        }else if(field_symmetry_==2){
         nk_=0;
         for(int ik=0;ik<nk_bz_;ik++){
                dvector tmp(2);
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                tmp(0)=(ix*1.0)*dk;
                tmp(1)=(iy*1.0)*dk;
                kpoints_bz_[ik]=tmp;
                kweight_bz_[ik]=1.0/nk_bz_;
                if(ix<=L_/2 && iy<=L_/2){
                    kpoints_[nk_]=tmp;
                    idx_kk_[ik]=nk_;
                    idx_inv_[nk_]=ik;
                    nk_++;
               }
             }
            for(int ik=0;ik<nk_bz_;ik++){
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                // map the point:
                if(ix>L_/2) ix=L_-ix;
                if(iy>L_/2) iy=L_-iy;
                int ik1=idx(ix,iy);
                idx_kk_[ik]=idx_kk_[ik1];
               }
           }else{
            nk_=0;
            for(int ik=0;ik<nk_bz_;ik++){
                dvector tmp(2);
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                tmp(0)=(ix*1.0)*dk;
                tmp(1)=(iy*1.0)*dk;
                kpoints_bz_[ik]=tmp;
                kweight_bz_[ik]=1.0/nk_bz_;
                if(ix>=iy){
                    kpoints_[nk_]=tmp;
                    idx_kk_[ik]=nk_;
                    idx_inv_[nk_]=ik;
                    nk_++;
                }
            }
            for(int ik=0;ik<nk_bz_;ik++){
                int ix=idx_ix(ik);
                int iy=idx_iy(ik);
                // map the point:
                if(ix<iy){
                    int itmp=ix;
                    ix=iy;
                    iy=itmp;
                }
                int ik1=idx(ix,iy);
                idx_kk_[ik]=idx_kk_[ik1];
            }
        }
   
        tx_.resize(nt_+2,1.0);
        ty_.resize(nt_+2,1.0);
        A_.resize(nt_+2,0.0);
        E_.resize(nt_+2,0.0);
        V_.resize(nt_+2,0.0);
        U0_.resize(nt_+2,0.0);
        
        mu_=0.0;
        nR_=(L_/2)*(L_/2); // I simple same all R-points in one quadrant
        Rpoints_.resize(nR_);
        for(int ix=0;ix<L_/2;ix++){
            for(int iy=0;iy<L_/2;iy++){
                int ir=ix*(L/2)+iy;
                Rpoints_[ir].resize(2);
                Rpoints_[ir](0)=ix;
                Rpoints_[ir](1)=iy;
            }
        }
        // a test: check that sum_{k,q} |k||k-q|=(sum_k |k|)^2
//        {
//            double sum1=0.0,sum2=0.0;
//            for(int k=0;k<nk_bz_;k++) sum1+=(kpoints_[k].dot(kpoints_[k]));
//            for(int q=0;q<nk_bz_;q++){
//                for(int k=0;k<nk_bz_;k++){
//                    int kq=add_kpoints_idx(k,q,-1);
//                    sum2+=(kpoints_[k].dot(kpoints_[k]))*(kpoints_[kq].dot(kpoints_[kq]));
//                }
//            }
//            std::cout << "sum1=" << sum1<<std::endl;
//            std::cout << "sum2=" << sum2<<std::endl;
//            std::cout << "sum1*sum1=" << sum1*sum1<<std::endl;
        
//          }
	}

//     cout << U0_[0] << endl;

    // (cx,cy)=(ax,ay)+(bx,by), all numbers integers on even mesh
    // return a+s*b
    int add_kpoints_1d(int a,int b,int s=1){
		int c=a+s*b;
        while (c<0){c += L_;}
		return (c%L_);
	}
    int add_kpoints_idx(int ia,int ib,int s=1){
		int ax=idx_ix(ia);
        int ay=idx_iy(ia);
        int bx=idx_ix(ib);
        int by=idx_iy(ib);
        int cx=add_kpoints_1d(ax,bx,s);
        int cy=add_kpoints_1d(ay,by,s);
        return idx(cx,cy);
	}
    void hk(cdmatrix &ek,int tstp, dvector kk){
        ek=cdmatrix(1,1);
        if((field_symmetry_==0) || (field_symmetry_==2)){
            ek(0,0)=-(2.0*tx_[tstp+1]*cos(kk(0)))-(2.0*ty_[tstp+1]*cos(kk(1)));
	    //if(tstp==-1){
            //ek(0,0)=-(2.0*tx_[tstp+1]*cos(kk(0)))-(2.0*ty_[tstp+1]*cos(kk(1))) + (1.20*(cos(kk(0))*cos(kk(1)))) ;
	   // }
	   // else{
	   // ek(0,0)=-(2.0*tx_[tstp+1]*cos(kk(0)))-(2.0*ty_[tstp+1]*cos(kk(1))) + (0.0*sin(30.0*tstp*0.0125)) ;	    
           // }	    
//         ek(0,0)=(-2.0*tx_[tstp+1]*(cos(kk(0))+cos(kk(1))))-(1.2*(cos(kk(0))*cos(kk(1))));
        }else if(field_symmetry_==1){
            ek(0,0)=-(2.0*tx_[tstp+1]*cos(kk(0)-A_[tstp+1]))-(2.0*ty_[tstp+1]*cos(kk(1)));
        }else{
          ek(0,0)=-(2.0*tx_[tstp+1]*cos(kk(0)-A_[tstp+1]))-(2.0*ty_[tstp+1]*cos(kk(1)-A_[tstp+1]));
        }
    }

    

    void V_qc(cdmatrix &ek,int tstp, dvector kk){
      ek=cdmatrix(1,1);
      ek(0,0)=(0.5*U0_[tstp+1])+(2.0*V_[tstp+1]*(cos(kk(0))+cos(kk(1))));  
    }

   void V_qz(cdmatrix &ek,int tstp, dvector kk){
      ek=cdmatrix(1,1);
      ek(0,0)=-(0.5*U0_[tstp+1]);
    }

   void Wk_qc(cdmatrix &ek,int tstp, dvector kk){
      ek=cdmatrix(1,1);
      ek(0,0)=(0.25*U0_[tstp+1])+(2.0*V_[tstp+1]*(cos(kk(0))+cos(kk(1))));
    }

   void Wk_qz(cdmatrix &ek,int tstp, dvector kk){
      ek=cdmatrix(1,1);
      ek(0,0)=-(0.25*U0_[tstp+1]);
    }

    void V_q(cdmatrix &ek,int tstp, dvector kk){
      ek=cdmatrix(1,1);
      ek(0,0)=(2.0*V_[tstp+1]*(cos(kk(0))+cos(kk(1))));
    }

    void vk(cdmatrix &vk,int tstp,dvector kk){
        vk=cdmatrix(1,1);
        if((field_symmetry_==0) || (field_symmetry_==2)){
            vk(0,0)= 0.0;
//          vk(0,0)=(2.0*tx_[tstp+1]*sin(kk(0)-A_[tstp+1]));
        }else if(field_symmetry_==1){
            vk(0,0)=(2.0*tx_[tstp+1]*sin(kk(0)-A_[tstp+1]));
        }else{
           vk(0,0)=(2.0*tx_[tstp+1]*sin(kk(0)-A_[tstp+1]))+(2.0*ty_[tstp+1]*sin(kk(1)-A_[tstp+1]));
	   //vk(0,0)=(2.0*tx_[tstp+1]*sin(kk(0)-A_[tstp+1]));
        } 
    }
     
    void dek(cdmatrix &vk, int tstp, dvector kk){
      vk=cdmatrix(1,1);
      vk(0,0)= (2.0*tx_[tstp+1]*sin(kk(0)-A_[tstp+1]));     
    }

    void dvk(cdmatrix &vk, int tstp, dvector kk){
      vk=cdmatrix(1,1);
      vk(0,0)= (2.0*tx_[tstp+1]*cos(kk(0)-A_[tstp+1]));
    }

};

} //namespace



#endif
