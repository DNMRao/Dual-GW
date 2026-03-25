#pragma once

#include <sys/stat.h>
#include <complex>
#include "cntr/cntr.hpp"
#include "Dgw_lattice.hpp"

using namespace cntr;

#define CFUNC cntr::function<double>
#define GREEN cntr::herm_matrix<double>
#define GREEN_TSTP cntr::herm_matrix_timestep<double>
#define CPLX std::complex<double>

namespace Dgw{

template <class LATT> class kpoint{
 public:
    void init(int nt,int ntau,int size,double beta,double h,dvector kk, LATT &latt);
    void init_observables();
    void set_hk(int tstp, LATT &latt);
    void set_vertex_c(int tstp, LATT &latt);
    void set_vertex_z(int tstp, LATT &latt);
    void set_vertex_Wk_c(int tstp, LATT &latt);
    void set_vertex_Wk_z(int tstp, LATT &latt);
    void set_hktilde(int tstp, LATT &latt);
    void set_vertex_(int tstp, LATT &latt);
    void Get_G_latt_DMFT(int tstp, int kt_, cntr::herm_matrix<double> &g_latt);
    void step_Wk_dual(int tstp, int kt_);
    void step_Wk_dual_V0(int tstp, int kt_,cntr::herm_matrix<double> &,cntr::herm_matrix<double> &);
    void step_Gk_dual(int tstp, int kt_);
    void step_Wk_dual_with_error(int tstp, int kt_, double &err1, double &err2);
    void step_Wk_dual_with_error_V0(int tstp, int kt_, double &err1, double &err2, 
		    cntr::herm_matrix<double> &, cntr::herm_matrix<double> &);
    void step_Gk_dual_with_error(int tstp, int kt_, double &err3);
    void Get_real_Glatt(int tstp, int kt_,cntr::herm_matrix<double> &g_loc_,
                 cntr::herm_matrix<double> &Delta_);
    void Get_real_chi_latt(int tstp, int kt_, 
               cntr::herm_matrix<double> &pi_c, cntr::herm_matrix<double> &pi_z, cntr::function<double> &U_);

    void Get_sigma_correction(int tstp, int kt_, cntr::herm_matrix<double> &g_loc_);
//    void Get_PE_energy(int tstp, int kt_, cntr::function<double> &PE_);
    double beta_;
    double h_;
    int nt_;
    int ntau_;
    int nrpa_; // Dimension of the Green's function

    CFUNC hk_, Vertex_c, Vertex_z, Vertex_Wk_c, Vertex_Wk_z, hktilde_;
    CFUNC Sigma_sg_, Vertex_, Hartree_,Sigma_sd_;
    GREEN Gk_DMFT, Gk_tilde, Wk_bare_c, Wk_bare_z; 
    GREEN Gk_dual,Wk_dual_c, Wk_dual_z, Pi_dual;
    GREEN Sigma_dual,Convo_1temp,Convo_2temp,Convo_1temp_cc,Convo_2temp_cc;
    GREEN Convo_3temp,Convo_3temp_cc;
    GREEN Gk_r,Chi_rc,Chi_rz,Pi_latt_z,Pi_latt_c;
    GREEN Pi_bar_c, Pi_bar_z,TK_bar,sc_c,sc_z,Pi_mix;
    GREEN Sigma_G,G_Sigma_G,G_Sigma_sg_G,Sigma_dual_mix,Sigma_G_cc;
    GREEN sigma_bar,Convo_4temp,Convo_4temp_cc;
    GREEN Sigma_D,Sigma_D_cc,G_Sigma_G_D,G_Sigma_G_D_cc,G_Sigma_sd_G;
    GREEN G_Si_G_D,G_Si_G_D_cc;   
   double mu_;
   dvector kk_;
   };
}
