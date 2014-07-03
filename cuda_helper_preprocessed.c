#include <stddef.h>
#include <stdio.h>
#include "cuda_helper.h"

#define NB_CUDA_KERNEL 42
int cuda_get_nb_kernels(void) {return NB_CUDA_KERNEL;}
void cuda_init_helper(void) {}
struct assemble_boundary_accel_on_device_args_s {
  const int *d_ibool_interfaces;
  const int *d_nibool_interfaces;
  const int max_nibool_interfaces;
  const int num_interfaces;
  const float *d_send_accel_buffer;
  float *d_accel;
};

struct assemble_boundary_potential_on_device_args_s {
  const int *d_ibool_interfaces;
  const int *d_nibool_interfaces;
  const int max_nibool_interfaces;
  const int num_interfaces;
  const float *d_send_potential_dot_dot_buffer;
  float *d_potential_dot_dot_acoustic;
};

struct compute_acoustic_kernel_args_s {
  const int NSPEC;
  const float deltat;
  float *kappa_ac_kl;
  float *rho_ac_kl;
  const float *b_potential_dot_dot_acoustic;
  const float *b_potential_acoustic;
  const float *potential_dot_dot_acoustic;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  const float *hprime_xx;
  const float *kappastore;
  const float *rhostore;
  const int *ibool;
};

struct compute_add_sources_adjoint_kernel_args_s {
  const int nadj_rec_local;
  const int *pre_computed_irec;
  const int *ispec_selected_rec;
  const int *ibool;
  const float *adj_sourcearrays;
  const int nrec;
  float *accel;
};

struct compute_add_sources_kernel_args_s {
  const int NSOURCES;
  const int *ispec_selected_source;
  const int *islice_selected_source;
  const int myrank;
  const double *stf_pre_compute;
  const float *sourcearrays;
  const int *ibool;
  float *accel;
};

struct compute_ani_kernel_args_s {
  const float deltat;
  const int NSPEC;
  float *cijkl_kl;
  const float *b_epsilon_trace_over_3;
  const float *b_epsilondev_yz;
  const float *b_epsilondev_xz;
  const float *b_epsilondev_xy;
  const float *b_epsilondev_yy;
  const float *b_epsilondev_xx;
  const float *epsilon_trace_over_3;
  const float *epsilondev_yz;
  const float *epsilondev_xz;
  const float *epsilondev_xy;
  const float *epsilondev_yy;
  const float *epsilondev_xx;
};

struct compute_ani_undo_att_kernel_args_s {
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  const float *d_b_displ;
  const int *d_ibool;
  const float deltat;
  const int NSPEC;
  float *cijkl_kl;
  const float *epsilon_trace_over_3;
  const float *epsilondev_yz;
  const float *epsilondev_xz;
  const float *epsilondev_xy;
  const float *epsilondev_yy;
  const float *epsilondev_xx;
};

struct compute_coupling_CMB_fluid_kernel_args_s {
  const int NSPEC2D_BOTTOM_CM;
  int GRAVITY;
  const float minus_g_cmb;
  const float RHO_TOP_OC;
  const int *ibelm_top_outer_core;
  const int *ibool_outer_core;
  const float *wgllwgll_xy;
  const float *jacobian2D_top_outer_core;
  const float *normal_top_outer_core;
  const int *ibelm_bottom_crust_mantle;
  const int *ibool_crust_mantle;
  const float *accel_outer_core;
  float *accel_crust_mantle;
  const float *displ_crust_mantle;
};

struct compute_coupling_ICB_fluid_kernel_args_s {
  const int NSPEC2D_TOP_IC;
  int GRAVITY;
  const float minus_g_icb;
  const float RHO_BOTTOM_OC;
  const int *ibelm_bottom_outer_core;
  const int *ibool_outer_core;
  const float *wgllwgll_xy;
  const float *jacobian2D_bottom_outer_core;
  const float *normal_bottom_outer_core;
  const int *ibelm_top_inner_core;
  const int *ibool_inner_core;
  const float *accel_outer_core;
  float *accel_inner_core;
  const float *displ_inner_core;
};

struct compute_coupling_fluid_CMB_kernel_args_s {
  const int NSPEC2D_TOP_OC;
  const int *ibelm_top_outer_core;
  const int *ibool_outer_core;
  const float *wgllwgll_xy;
  const float *jacobian2D_top_outer_core;
  const float *normal_top_outer_core;
  const int *ibelm_bottom_crust_mantle;
  const int *ibool_crust_mantle;
  float *accel_outer_core;
  const float *displ_crust_mantle;
};

struct compute_coupling_fluid_ICB_kernel_args_s {
  const int NSPEC2D_BOTTOM_OC;
  const int *ibelm_bottom_outer_core;
  const int *ibool_outer_core;
  const float *wgllwgll_xy;
  const float *jacobian2D_bottom_outer_core;
  const float *normal_bottom_outer_core;
  const int *ibelm_top_inner_core;
  const int *ibool_inner_core;
  float *accel_outer_core;
  const float *displ_inner_core;
};

struct compute_coupling_ocean_kernel_args_s {
  const float *normal_ocean_load;
  const int *ibool_ocean_load;
  const int npoin_ocean_load;
  const float *rmass_ocean_load;
  const float *rmassz_crust_mantle;
  const float *rmassy_crust_mantle;
  const float *rmassx_crust_mantle;
  float *accel_crust_mantle;
};

struct compute_hess_kernel_args_s {
  const int NSPEC_AB;
  const float deltat;
  float *hess_kl;
  const float *b_accel;
  const float *accel;
  const int *ibool;
};

struct compute_iso_kernel_args_s {
  const float deltat;
  const int NSPEC;
  float *kappa_kl;
  float *mu_kl;
  const float *b_epsilon_trace_over_3;
  const float *b_epsilondev_yz;
  const float *b_epsilondev_xz;
  const float *b_epsilondev_xy;
  const float *b_epsilondev_yy;
  const float *b_epsilondev_xx;
  const float *epsilon_trace_over_3;
  const float *epsilondev_yz;
  const float *epsilondev_xz;
  const float *epsilondev_xy;
  const float *epsilondev_yy;
  const float *epsilondev_xx;
};

struct compute_iso_undo_att_kernel_args_s {
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  const float *d_b_displ;
  const int *d_ibool;
  const float deltat;
  const int NSPEC;
  float *kappa_kl;
  float *mu_kl;
  const float *epsilon_trace_over_3;
  const float *epsilondev_yz;
  const float *epsilondev_xz;
  const float *epsilondev_xy;
  const float *epsilondev_yy;
  const float *epsilondev_xx;
};

struct compute_rho_kernel_args_s {
  const float deltat;
  const int NSPEC;
  float *rho_kl;
  const float *b_displ;
  const float *accel;
  const int *ibool;
};

struct compute_stacey_acoustic_backward_kernel_args_s {
  const int *ibool;
  const int *nimax;
  const int *nimin;
  const int *njmax;
  const int *njmin;
  const int *nkmin_eta;
  const int *nkmin_xi;
  const int *abs_boundary_ispec;
  const int num_abs_boundary_faces;
  const int interface_type;
  const float *b_absorb_potential;
  float *b_potential_dot_dot_acoustic;
};

struct compute_stacey_acoustic_kernel_args_s {
  float *b_absorb_potential;
  const int SAVE_FORWARD;
  const float *vpstore;
  const int *ibool;
  const float *wgllwgll;
  const float *abs_boundary_jacobian2D;
  const int *nimax;
  const int *nimin;
  const int *njmax;
  const int *njmin;
  const int *nkmin_eta;
  const int *nkmin_xi;
  const int *abs_boundary_ispec;
  const int num_abs_boundary_faces;
  const int interface_type;
  float *potential_dot_dot_acoustic;
  const float *potential_dot_acoustic;
};

struct compute_stacey_elastic_backward_kernel_args_s {
  const int *ibool;
  const int *nimax;
  const int *nimin;
  const int *njmax;
  const int *njmin;
  const int *nkmin_eta;
  const int *nkmin_xi;
  const int *abs_boundary_ispec;
  const int num_abs_boundary_faces;
  const int interface_type;
  const float *b_absorb_field;
  float *b_accel;
};

struct compute_stacey_elastic_kernel_args_s {
  float *b_absorb_field;
  const int SAVE_FORWARD;
  const float *rho_vs;
  const float *rho_vp;
  const int *ibool;
  const float *wgllwgll;
  const float *abs_boundary_jacobian2D;
  const float *abs_boundary_normal;
  const int *nimax;
  const int *nimin;
  const int *njmax;
  const int *njmin;
  const int *nkmin_eta;
  const int *nkmin_xi;
  const int *abs_boundary_ispec;
  const int num_abs_boundary_faces;
  const int interface_type;
  float *accel;
  const float *veloc;
};

struct compute_strength_noise_kernel_args_s {
  const int nspec_top;
  const float deltat;
  float *Sigma_kl;
  const float *normal_z_noise;
  const float *normal_y_noise;
  const float *normal_x_noise;
  const float *noise_surface_movie;
  const int *ibool;
  const int *ibelm_top;
  const float *displ;
};

struct crust_mantle_impl_kernel_adjoint_args_s {
  const int NSPEC_CRUST_MANTLE_STRAIN_ONLY;
  const float *wgll_cube;
  const float *d_density_table;
  const float *d_minus_deriv_gravity_table;
  const float *d_minus_gravity_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *d_c66store;
  const float *d_c56store;
  const float *d_c55store;
  const float *d_c46store;
  const float *d_c45store;
  const float *d_c44store;
  const float *d_c36store;
  const float *d_c35store;
  const float *d_c34store;
  const float *d_c33store;
  const float *d_c26store;
  const float *d_c25store;
  const float *d_c24store;
  const float *d_c23store;
  const float *d_c22store;
  const float *d_c16store;
  const float *d_c15store;
  const float *d_c14store;
  const float *d_c13store;
  const float *d_c12store;
  const float *d_c11store;
  const int ANISOTROPY;
  const float *gammaval;
  const float *betaval;
  const float *alphaval;
  float *R_yz;
  float *R_xz;
  float *R_xy;
  float *R_yy;
  float *R_xx;
  const float *factor_common;
  const float *one_minus_sum_beta;
  const int USE_3D_ATTENUATION_ARRAYS;
  const int PARTIAL_PHYS_DISPERSION_ONLY;
  const int ATTENUATION;
  float *epsilon_trace_over_3;
  float *epsilondev_yz;
  float *epsilondev_xz;
  float *epsilondev_xy;
  float *epsilondev_yy;
  float *epsilondev_xx;
  const int COMPUTE_AND_STORE_STRAIN;
  const float *d_eta_anisostore;
  const float *d_muhstore;
  const float *d_kappahstore;
  const float *d_muvstore;
  const float *d_kappavstore;
  const float *d_wgllwgll_yz;
  const float *d_wgllwgll_xz;
  const float *d_wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_accel;
  const float *d_displ;
  const int use_mesh_coloring_gpu;
  const float deltat;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_ispec_is_tiso;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct crust_mantle_impl_kernel_forward_args_s {
  const int NSPEC_CRUST_MANTLE_STRAIN_ONLY;
  const float *wgll_cube;
  const float *d_density_table;
  const float *d_minus_deriv_gravity_table;
  const float *d_minus_gravity_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *d_c66store;
  const float *d_c56store;
  const float *d_c55store;
  const float *d_c46store;
  const float *d_c45store;
  const float *d_c44store;
  const float *d_c36store;
  const float *d_c35store;
  const float *d_c34store;
  const float *d_c33store;
  const float *d_c26store;
  const float *d_c25store;
  const float *d_c24store;
  const float *d_c23store;
  const float *d_c22store;
  const float *d_c16store;
  const float *d_c15store;
  const float *d_c14store;
  const float *d_c13store;
  const float *d_c12store;
  const float *d_c11store;
  const int ANISOTROPY;
  const float *gammaval;
  const float *betaval;
  const float *alphaval;
  float *R_yz;
  float *R_xz;
  float *R_xy;
  float *R_yy;
  float *R_xx;
  const float *factor_common;
  const float *one_minus_sum_beta;
  const int USE_3D_ATTENUATION_ARRAYS;
  const int PARTIAL_PHYS_DISPERSION_ONLY;
  const int ATTENUATION;
  float *epsilon_trace_over_3;
  float *epsilondev_yz;
  float *epsilondev_xz;
  float *epsilondev_xy;
  float *epsilondev_yy;
  float *epsilondev_xx;
  const int COMPUTE_AND_STORE_STRAIN;
  const float *d_eta_anisostore;
  const float *d_muhstore;
  const float *d_kappahstore;
  const float *d_muvstore;
  const float *d_kappavstore;
  const float *d_wgllwgll_yz;
  const float *d_wgllwgll_xz;
  const float *d_wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_accel;
  const float *d_displ;
  const int use_mesh_coloring_gpu;
  const float deltat;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_ispec_is_tiso;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct get_maximum_scalar_kernel_args_s {
  float *d_max;
  const int size;
  const float *array;
};

struct get_maximum_vector_kernel_args_s {
  float *d_max;
  const int size;
  const float *array;
};

struct inner_core_impl_kernel_adjoint_args_s {
  const int NSPEC_INNER_CORE;
  const int NSPEC_INNER_CORE_STRAIN_ONLY;
  const float *wgll_cube;
  const float *d_density_table;
  const float *d_minus_deriv_gravity_table;
  const float *d_minus_gravity_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *d_c44store;
  const float *d_c33store;
  const float *d_c13store;
  const float *d_c12store;
  const float *d_c11store;
  const int ANISOTROPY;
  const float *gammaval;
  const float *betaval;
  const float *alphaval;
  float *R_yz;
  float *R_xz;
  float *R_xy;
  float *R_yy;
  float *R_xx;
  const float *factor_common;
  const float *one_minus_sum_beta;
  const int USE_3D_ATTENUATION_ARRAYS;
  const int PARTIAL_PHYS_DISPERSION_ONLY;
  const int ATTENUATION;
  float *epsilon_trace_over_3;
  float *epsilondev_yz;
  float *epsilondev_xz;
  float *epsilondev_xy;
  float *epsilondev_yy;
  float *epsilondev_xx;
  const int COMPUTE_AND_STORE_STRAIN;
  const float *d_muvstore;
  const float *d_kappavstore;
  const float *d_wgllwgll_yz;
  const float *d_wgllwgll_xz;
  const float *d_wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_accel;
  const float *d_displ;
  const int use_mesh_coloring_gpu;
  const float deltat;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_idoubling;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct inner_core_impl_kernel_forward_args_s {
  const int NSPEC_INNER_CORE;
  const int NSPEC_INNER_CORE_STRAIN_ONLY;
  const float *wgll_cube;
  const float *d_density_table;
  const float *d_minus_deriv_gravity_table;
  const float *d_minus_gravity_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *d_c44store;
  const float *d_c33store;
  const float *d_c13store;
  const float *d_c12store;
  const float *d_c11store;
  const int ANISOTROPY;
  const float *gammaval;
  const float *betaval;
  const float *alphaval;
  float *R_yz;
  float *R_xz;
  float *R_xy;
  float *R_yy;
  float *R_xx;
  const float *factor_common;
  const float *one_minus_sum_beta;
  const int USE_3D_ATTENUATION_ARRAYS;
  const int PARTIAL_PHYS_DISPERSION_ONLY;
  const int ATTENUATION;
  float *epsilon_trace_over_3;
  float *epsilondev_yz;
  float *epsilondev_xz;
  float *epsilondev_xy;
  float *epsilondev_yy;
  float *epsilondev_xx;
  const int COMPUTE_AND_STORE_STRAIN;
  const float *d_muvstore;
  const float *d_kappavstore;
  const float *d_wgllwgll_yz;
  const float *d_wgllwgll_xz;
  const float *d_wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_accel;
  const float *d_displ;
  const int use_mesh_coloring_gpu;
  const float deltat;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_idoubling;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct noise_add_source_master_rec_kernel_args_s {
  const int it;
  const float *noise_sourcearray;
  float *accel;
  const int irec_master_noise;
  const int *ispec_selected_rec;
  const int *ibool;
};

struct noise_add_surface_movie_kernel_args_s {
  const float *wgllwgll;
  const float *jacobian2D;
  const float *mask_noise;
  const float *normal_z_noise;
  const float *normal_y_noise;
  const float *normal_x_noise;
  const float *noise_surface_movie;
  const int nspec_top;
  const int *ibelm_top;
  const int *ibool;
  float *accel;
};

struct noise_transfer_surface_to_host_kernel_args_s {
  float *noise_surface_movie;
  const float *displ;
  const int *ibool;
  const int nspec_top;
  const int *ibelm_top;
};

struct outer_core_impl_kernel_adjoint_args_s {
  const int NSPEC_OUTER_CORE;
  float *d_B_array_rotation;
  float *d_A_array_rotation;
  const float deltat;
  const float two_omega_earth;
  const float time;
  const int ROTATION;
  const float *wgll_cube;
  const float *d_minus_rho_g_over_kappa_fluid;
  const float *d_d_ln_density_dr_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *wgllwgll_yz;
  const float *wgllwgll_xz;
  const float *wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_potential_dot_dot;
  const float *d_potential;
  const int use_mesh_coloring_gpu;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct outer_core_impl_kernel_forward_args_s {
  const int NSPEC_OUTER_CORE;
  float *d_B_array_rotation;
  float *d_A_array_rotation;
  const float deltat;
  const float two_omega_earth;
  const float time;
  const int ROTATION;
  const float *wgll_cube;
  const float *d_minus_rho_g_over_kappa_fluid;
  const float *d_d_ln_density_dr_table;
  const float *d_zstore;
  const float *d_ystore;
  const float *d_xstore;
  const int GRAVITY;
  const float *wgllwgll_yz;
  const float *wgllwgll_xz;
  const float *wgllwgll_xy;
  const float *d_hprimewgll_xx;
  const float *d_hprime_xx;
  const float *d_gammaz;
  const float *d_gammay;
  const float *d_gammax;
  const float *d_etaz;
  const float *d_etay;
  const float *d_etax;
  const float *d_xiz;
  const float *d_xiy;
  const float *d_xix;
  float *d_potential_dot_dot;
  const float *d_potential;
  const int use_mesh_coloring_gpu;
  const int d_iphase;
  const int num_phase_ispec;
  const int *d_phase_ispec_inner;
  const int *d_ibool;
  const int nb_blocks_to_compute;
};

struct prepare_boundary_accel_on_device_args_s {
  const int *d_ibool_interfaces;
  const int *d_nibool_interfaces;
  const int max_nibool_interfaces;
  const int num_interfaces;
  float *d_send_accel_buffer;
  const float *d_accel;
};

struct prepare_boundary_potential_on_device_args_s {
  const int *d_ibool_interfaces;
  const int *d_nibool_interfaces;
  const int max_nibool_interfaces;
  const int num_interfaces;
  float *d_send_potential_dot_dot_buffer;
  const float *d_potential_dot_dot_acoustic;
};

struct update_accel_acoustic_kernel_args_s {
  const float *rmass;
  const int size;
  float *accel;
};

struct update_accel_elastic_kernel_args_s {
  const float *rmassz;
  const float *rmassy;
  const float *rmassx;
  const float two_omega_earth;
  const int size;
  const float *veloc;
  float *accel;
};

struct update_disp_veloc_kernel_args_s {
  const float deltatover2;
  const float deltatsqover2;
  const float deltat;
  const int size;
  float *accel;
  float *veloc;
  float *displ;
};

struct update_potential_kernel_args_s {
  const float deltatover2;
  const float deltatsqover2;
  const float deltat;
  const int size;
  float *potential_dot_dot_acoustic;
  float *potential_dot_acoustic;
  float *potential_acoustic;
};

struct update_veloc_acoustic_kernel_args_s {
  const float deltatover2;
  const int size;
  const float *accel;
  float *veloc;
};

struct update_veloc_elastic_kernel_args_s {
  const float deltatover2;
  const int size;
  const float *accel;
  float *veloc;
};

struct write_seismograms_transfer_from_device_kernel_args_s {
  const int nrec_local;
  const float *d_field;
  float *station_seismo_field;
  const int *ibool;
  const int *ispec_selected_rec;
  const int *number_receiver_global;
};

struct write_seismograms_transfer_strain_from_device_kernel_args_s {
  const int nrec_local;
  const float *d_field;
  float *station_strain_field;
  const int *ibool;
  const int *ispec_selected_rec;
  const int *number_receiver_global;
};

struct param_info_s assemble_boundary_accel_on_device_params[] = {
  {"d_accel", "float *"},
  {"d_send_accel_buffer", "const float *"},
  {"num_interfaces", "const int"},
  {"max_nibool_interfaces", "const int"},
  {"d_nibool_interfaces", "const int *"},
  {"d_ibool_interfaces", "const int *"}
};

struct param_info_s assemble_boundary_potential_on_device_params[] = {
  {"d_potential_dot_dot_acoustic", "float *"},
  {"d_send_potential_dot_dot_buffer", "const float *"},
  {"num_interfaces", "const int"},
  {"max_nibool_interfaces", "const int"},
  {"d_nibool_interfaces", "const int *"},
  {"d_ibool_interfaces", "const int *"}
};

struct param_info_s compute_acoustic_kernel_params[] = {
  {"ibool", "const int *"},
  {"rhostore", "const float *"},
  {"kappastore", "const float *"},
  {"hprime_xx", "const float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"potential_dot_dot_acoustic", "const float *"},
  {"b_potential_acoustic", "const float *"},
  {"b_potential_dot_dot_acoustic", "const float *"},
  {"rho_ac_kl", "float *"},
  {"kappa_ac_kl", "float *"},
  {"deltat", "const float"},
  {"NSPEC", "const int"}
};

struct param_info_s compute_add_sources_adjoint_kernel_params[] = {
  {"accel", "float *"},
  {"nrec", "const int"},
  {"adj_sourcearrays", "const float *"},
  {"ibool", "const int *"},
  {"ispec_selected_rec", "const int *"},
  {"pre_computed_irec", "const int *"},
  {"nadj_rec_local", "const int"}
};

struct param_info_s compute_add_sources_kernel_params[] = {
  {"accel", "float *"},
  {"ibool", "const int *"},
  {"sourcearrays", "const float *"},
  {"stf_pre_compute", "const double *"},
  {"myrank", "const int"},
  {"islice_selected_source", "const int *"},
  {"ispec_selected_source", "const int *"},
  {"NSOURCES", "const int"}
};

struct param_info_s compute_ani_kernel_params[] = {
  {"epsilondev_xx", "const float *"},
  {"epsilondev_yy", "const float *"},
  {"epsilondev_xy", "const float *"},
  {"epsilondev_xz", "const float *"},
  {"epsilondev_yz", "const float *"},
  {"epsilon_trace_over_3", "const float *"},
  {"b_epsilondev_xx", "const float *"},
  {"b_epsilondev_yy", "const float *"},
  {"b_epsilondev_xy", "const float *"},
  {"b_epsilondev_xz", "const float *"},
  {"b_epsilondev_yz", "const float *"},
  {"b_epsilon_trace_over_3", "const float *"},
  {"cijkl_kl", "float *"},
  {"NSPEC", "const int"},
  {"deltat", "const float"}
};

struct param_info_s compute_ani_undo_att_kernel_params[] = {
  {"epsilondev_xx", "const float *"},
  {"epsilondev_yy", "const float *"},
  {"epsilondev_xy", "const float *"},
  {"epsilondev_xz", "const float *"},
  {"epsilondev_yz", "const float *"},
  {"epsilon_trace_over_3", "const float *"},
  {"cijkl_kl", "float *"},
  {"NSPEC", "const int"},
  {"deltat", "const float"},
  {"d_ibool", "const int *"},
  {"d_b_displ", "const float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"}
};

struct param_info_s compute_coupling_CMB_fluid_kernel_params[] = {
  {"displ_crust_mantle", "const float *"},
  {"accel_crust_mantle", "float *"},
  {"accel_outer_core", "const float *"},
  {"ibool_crust_mantle", "const int *"},
  {"ibelm_bottom_crust_mantle", "const int *"},
  {"normal_top_outer_core", "const float *"},
  {"jacobian2D_top_outer_core", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"ibool_outer_core", "const int *"},
  {"ibelm_top_outer_core", "const int *"},
  {"RHO_TOP_OC", "const float"},
  {"minus_g_cmb", "const float"},
  {"GRAVITY", "int"},
  {"NSPEC2D_BOTTOM_CM", "const int"}
};

struct param_info_s compute_coupling_ICB_fluid_kernel_params[] = {
  {"displ_inner_core", "const float *"},
  {"accel_inner_core", "float *"},
  {"accel_outer_core", "const float *"},
  {"ibool_inner_core", "const int *"},
  {"ibelm_top_inner_core", "const int *"},
  {"normal_bottom_outer_core", "const float *"},
  {"jacobian2D_bottom_outer_core", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"ibool_outer_core", "const int *"},
  {"ibelm_bottom_outer_core", "const int *"},
  {"RHO_BOTTOM_OC", "const float"},
  {"minus_g_icb", "const float"},
  {"GRAVITY", "int"},
  {"NSPEC2D_TOP_IC", "const int"}
};

struct param_info_s compute_coupling_fluid_CMB_kernel_params[] = {
  {"displ_crust_mantle", "const float *"},
  {"accel_outer_core", "float *"},
  {"ibool_crust_mantle", "const int *"},
  {"ibelm_bottom_crust_mantle", "const int *"},
  {"normal_top_outer_core", "const float *"},
  {"jacobian2D_top_outer_core", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"ibool_outer_core", "const int *"},
  {"ibelm_top_outer_core", "const int *"},
  {"NSPEC2D_TOP_OC", "const int"}
};

struct param_info_s compute_coupling_fluid_ICB_kernel_params[] = {
  {"displ_inner_core", "const float *"},
  {"accel_outer_core", "float *"},
  {"ibool_inner_core", "const int *"},
  {"ibelm_top_inner_core", "const int *"},
  {"normal_bottom_outer_core", "const float *"},
  {"jacobian2D_bottom_outer_core", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"ibool_outer_core", "const int *"},
  {"ibelm_bottom_outer_core", "const int *"},
  {"NSPEC2D_BOTTOM_OC", "const int"}
};

struct param_info_s compute_coupling_ocean_kernel_params[] = {
  {"accel_crust_mantle", "float *"},
  {"rmassx_crust_mantle", "const float *"},
  {"rmassy_crust_mantle", "const float *"},
  {"rmassz_crust_mantle", "const float *"},
  {"rmass_ocean_load", "const float *"},
  {"npoin_ocean_load", "const int"},
  {"ibool_ocean_load", "const int *"},
  {"normal_ocean_load", "const float *"}
};

struct param_info_s compute_hess_kernel_params[] = {
  {"ibool", "const int *"},
  {"accel", "const float *"},
  {"b_accel", "const float *"},
  {"hess_kl", "float *"},
  {"deltat", "const float"},
  {"NSPEC_AB", "const int"}
};

struct param_info_s compute_iso_kernel_params[] = {
  {"epsilondev_xx", "const float *"},
  {"epsilondev_yy", "const float *"},
  {"epsilondev_xy", "const float *"},
  {"epsilondev_xz", "const float *"},
  {"epsilondev_yz", "const float *"},
  {"epsilon_trace_over_3", "const float *"},
  {"b_epsilondev_xx", "const float *"},
  {"b_epsilondev_yy", "const float *"},
  {"b_epsilondev_xy", "const float *"},
  {"b_epsilondev_xz", "const float *"},
  {"b_epsilondev_yz", "const float *"},
  {"b_epsilon_trace_over_3", "const float *"},
  {"mu_kl", "float *"},
  {"kappa_kl", "float *"},
  {"NSPEC", "const int"},
  {"deltat", "const float"}
};

struct param_info_s compute_iso_undo_att_kernel_params[] = {
  {"epsilondev_xx", "const float *"},
  {"epsilondev_yy", "const float *"},
  {"epsilondev_xy", "const float *"},
  {"epsilondev_xz", "const float *"},
  {"epsilondev_yz", "const float *"},
  {"epsilon_trace_over_3", "const float *"},
  {"mu_kl", "float *"},
  {"kappa_kl", "float *"},
  {"NSPEC", "const int"},
  {"deltat", "const float"},
  {"d_ibool", "const int *"},
  {"d_b_displ", "const float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"}
};

struct param_info_s compute_rho_kernel_params[] = {
  {"ibool", "const int *"},
  {"accel", "const float *"},
  {"b_displ", "const float *"},
  {"rho_kl", "float *"},
  {"NSPEC", "const int"},
  {"deltat", "const float"}
};

struct param_info_s compute_stacey_acoustic_backward_kernel_params[] = {
  {"b_potential_dot_dot_acoustic", "float *"},
  {"b_absorb_potential", "const float *"},
  {"interface_type", "const int"},
  {"num_abs_boundary_faces", "const int"},
  {"abs_boundary_ispec", "const int *"},
  {"nkmin_xi", "const int *"},
  {"nkmin_eta", "const int *"},
  {"njmin", "const int *"},
  {"njmax", "const int *"},
  {"nimin", "const int *"},
  {"nimax", "const int *"},
  {"ibool", "const int *"}
};

struct param_info_s compute_stacey_acoustic_kernel_params[] = {
  {"potential_dot_acoustic", "const float *"},
  {"potential_dot_dot_acoustic", "float *"},
  {"interface_type", "const int"},
  {"num_abs_boundary_faces", "const int"},
  {"abs_boundary_ispec", "const int *"},
  {"nkmin_xi", "const int *"},
  {"nkmin_eta", "const int *"},
  {"njmin", "const int *"},
  {"njmax", "const int *"},
  {"nimin", "const int *"},
  {"nimax", "const int *"},
  {"abs_boundary_jacobian2D", "const float *"},
  {"wgllwgll", "const float *"},
  {"ibool", "const int *"},
  {"vpstore", "const float *"},
  {"SAVE_FORWARD", "const int"},
  {"b_absorb_potential", "float *"}
};

struct param_info_s compute_stacey_elastic_backward_kernel_params[] = {
  {"b_accel", "float *"},
  {"b_absorb_field", "const float *"},
  {"interface_type", "const int"},
  {"num_abs_boundary_faces", "const int"},
  {"abs_boundary_ispec", "const int *"},
  {"nkmin_xi", "const int *"},
  {"nkmin_eta", "const int *"},
  {"njmin", "const int *"},
  {"njmax", "const int *"},
  {"nimin", "const int *"},
  {"nimax", "const int *"},
  {"ibool", "const int *"}
};

struct param_info_s compute_stacey_elastic_kernel_params[] = {
  {"veloc", "const float *"},
  {"accel", "float *"},
  {"interface_type", "const int"},
  {"num_abs_boundary_faces", "const int"},
  {"abs_boundary_ispec", "const int *"},
  {"nkmin_xi", "const int *"},
  {"nkmin_eta", "const int *"},
  {"njmin", "const int *"},
  {"njmax", "const int *"},
  {"nimin", "const int *"},
  {"nimax", "const int *"},
  {"abs_boundary_normal", "const float *"},
  {"abs_boundary_jacobian2D", "const float *"},
  {"wgllwgll", "const float *"},
  {"ibool", "const int *"},
  {"rho_vp", "const float *"},
  {"rho_vs", "const float *"},
  {"SAVE_FORWARD", "const int"},
  {"b_absorb_field", "float *"}
};

struct param_info_s compute_strength_noise_kernel_params[] = {
  {"displ", "const float *"},
  {"ibelm_top", "const int *"},
  {"ibool", "const int *"},
  {"noise_surface_movie", "const float *"},
  {"normal_x_noise", "const float *"},
  {"normal_y_noise", "const float *"},
  {"normal_z_noise", "const float *"},
  {"Sigma_kl", "float *"},
  {"deltat", "const float"},
  {"nspec_top", "const int"}
};

struct param_info_s crust_mantle_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_ispec_is_tiso", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"deltat", "const float"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_displ", "const float *"},
  {"d_accel", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"d_wgllwgll_xy", "const float *"},
  {"d_wgllwgll_xz", "const float *"},
  {"d_wgllwgll_yz", "const float *"},
  {"d_kappavstore", "const float *"},
  {"d_muvstore", "const float *"},
  {"d_kappahstore", "const float *"},
  {"d_muhstore", "const float *"},
  {"d_eta_anisostore", "const float *"},
  {"COMPUTE_AND_STORE_STRAIN", "const int"},
  {"epsilondev_xx", "float *"},
  {"epsilondev_yy", "float *"},
  {"epsilondev_xy", "float *"},
  {"epsilondev_xz", "float *"},
  {"epsilondev_yz", "float *"},
  {"epsilon_trace_over_3", "float *"},
  {"ATTENUATION", "const int"},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int"},
  {"USE_3D_ATTENUATION_ARRAYS", "const int"},
  {"one_minus_sum_beta", "const float *"},
  {"factor_common", "const float *"},
  {"R_xx", "float *"},
  {"R_yy", "float *"},
  {"R_xy", "float *"},
  {"R_xz", "float *"},
  {"R_yz", "float *"},
  {"alphaval", "const float *"},
  {"betaval", "const float *"},
  {"gammaval", "const float *"},
  {"ANISOTROPY", "const int"},
  {"d_c11store", "const float *"},
  {"d_c12store", "const float *"},
  {"d_c13store", "const float *"},
  {"d_c14store", "const float *"},
  {"d_c15store", "const float *"},
  {"d_c16store", "const float *"},
  {"d_c22store", "const float *"},
  {"d_c23store", "const float *"},
  {"d_c24store", "const float *"},
  {"d_c25store", "const float *"},
  {"d_c26store", "const float *"},
  {"d_c33store", "const float *"},
  {"d_c34store", "const float *"},
  {"d_c35store", "const float *"},
  {"d_c36store", "const float *"},
  {"d_c44store", "const float *"},
  {"d_c45store", "const float *"},
  {"d_c46store", "const float *"},
  {"d_c55store", "const float *"},
  {"d_c56store", "const float *"},
  {"d_c66store", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_minus_gravity_table", "const float *"},
  {"d_minus_deriv_gravity_table", "const float *"},
  {"d_density_table", "const float *"},
  {"wgll_cube", "const float *"},
  {"NSPEC_CRUST_MANTLE_STRAIN_ONLY", "const int"}
};

struct param_info_s crust_mantle_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_ispec_is_tiso", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"deltat", "const float"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_displ", "const float *"},
  {"d_accel", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"d_wgllwgll_xy", "const float *"},
  {"d_wgllwgll_xz", "const float *"},
  {"d_wgllwgll_yz", "const float *"},
  {"d_kappavstore", "const float *"},
  {"d_muvstore", "const float *"},
  {"d_kappahstore", "const float *"},
  {"d_muhstore", "const float *"},
  {"d_eta_anisostore", "const float *"},
  {"COMPUTE_AND_STORE_STRAIN", "const int"},
  {"epsilondev_xx", "float *"},
  {"epsilondev_yy", "float *"},
  {"epsilondev_xy", "float *"},
  {"epsilondev_xz", "float *"},
  {"epsilondev_yz", "float *"},
  {"epsilon_trace_over_3", "float *"},
  {"ATTENUATION", "const int"},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int"},
  {"USE_3D_ATTENUATION_ARRAYS", "const int"},
  {"one_minus_sum_beta", "const float *"},
  {"factor_common", "const float *"},
  {"R_xx", "float *"},
  {"R_yy", "float *"},
  {"R_xy", "float *"},
  {"R_xz", "float *"},
  {"R_yz", "float *"},
  {"alphaval", "const float *"},
  {"betaval", "const float *"},
  {"gammaval", "const float *"},
  {"ANISOTROPY", "const int"},
  {"d_c11store", "const float *"},
  {"d_c12store", "const float *"},
  {"d_c13store", "const float *"},
  {"d_c14store", "const float *"},
  {"d_c15store", "const float *"},
  {"d_c16store", "const float *"},
  {"d_c22store", "const float *"},
  {"d_c23store", "const float *"},
  {"d_c24store", "const float *"},
  {"d_c25store", "const float *"},
  {"d_c26store", "const float *"},
  {"d_c33store", "const float *"},
  {"d_c34store", "const float *"},
  {"d_c35store", "const float *"},
  {"d_c36store", "const float *"},
  {"d_c44store", "const float *"},
  {"d_c45store", "const float *"},
  {"d_c46store", "const float *"},
  {"d_c55store", "const float *"},
  {"d_c56store", "const float *"},
  {"d_c66store", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_minus_gravity_table", "const float *"},
  {"d_minus_deriv_gravity_table", "const float *"},
  {"d_density_table", "const float *"},
  {"wgll_cube", "const float *"},
  {"NSPEC_CRUST_MANTLE_STRAIN_ONLY", "const int"}
};

struct param_info_s get_maximum_scalar_kernel_params[] = {
  {"array", "const float *"},
  {"size", "const int"},
  {"d_max", "float *"}
};

struct param_info_s get_maximum_vector_kernel_params[] = {
  {"array", "const float *"},
  {"size", "const int"},
  {"d_max", "float *"}
};

struct param_info_s inner_core_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_idoubling", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"deltat", "const float"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_displ", "const float *"},
  {"d_accel", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"d_wgllwgll_xy", "const float *"},
  {"d_wgllwgll_xz", "const float *"},
  {"d_wgllwgll_yz", "const float *"},
  {"d_kappavstore", "const float *"},
  {"d_muvstore", "const float *"},
  {"COMPUTE_AND_STORE_STRAIN", "const int"},
  {"epsilondev_xx", "float *"},
  {"epsilondev_yy", "float *"},
  {"epsilondev_xy", "float *"},
  {"epsilondev_xz", "float *"},
  {"epsilondev_yz", "float *"},
  {"epsilon_trace_over_3", "float *"},
  {"ATTENUATION", "const int"},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int"},
  {"USE_3D_ATTENUATION_ARRAYS", "const int"},
  {"one_minus_sum_beta", "const float *"},
  {"factor_common", "const float *"},
  {"R_xx", "float *"},
  {"R_yy", "float *"},
  {"R_xy", "float *"},
  {"R_xz", "float *"},
  {"R_yz", "float *"},
  {"alphaval", "const float *"},
  {"betaval", "const float *"},
  {"gammaval", "const float *"},
  {"ANISOTROPY", "const int"},
  {"d_c11store", "const float *"},
  {"d_c12store", "const float *"},
  {"d_c13store", "const float *"},
  {"d_c33store", "const float *"},
  {"d_c44store", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_minus_gravity_table", "const float *"},
  {"d_minus_deriv_gravity_table", "const float *"},
  {"d_density_table", "const float *"},
  {"wgll_cube", "const float *"},
  {"NSPEC_INNER_CORE_STRAIN_ONLY", "const int"},
  {"NSPEC_INNER_CORE", "const int"}
};

struct param_info_s inner_core_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_idoubling", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"deltat", "const float"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_displ", "const float *"},
  {"d_accel", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"d_wgllwgll_xy", "const float *"},
  {"d_wgllwgll_xz", "const float *"},
  {"d_wgllwgll_yz", "const float *"},
  {"d_kappavstore", "const float *"},
  {"d_muvstore", "const float *"},
  {"COMPUTE_AND_STORE_STRAIN", "const int"},
  {"epsilondev_xx", "float *"},
  {"epsilondev_yy", "float *"},
  {"epsilondev_xy", "float *"},
  {"epsilondev_xz", "float *"},
  {"epsilondev_yz", "float *"},
  {"epsilon_trace_over_3", "float *"},
  {"ATTENUATION", "const int"},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int"},
  {"USE_3D_ATTENUATION_ARRAYS", "const int"},
  {"one_minus_sum_beta", "const float *"},
  {"factor_common", "const float *"},
  {"R_xx", "float *"},
  {"R_yy", "float *"},
  {"R_xy", "float *"},
  {"R_xz", "float *"},
  {"R_yz", "float *"},
  {"alphaval", "const float *"},
  {"betaval", "const float *"},
  {"gammaval", "const float *"},
  {"ANISOTROPY", "const int"},
  {"d_c11store", "const float *"},
  {"d_c12store", "const float *"},
  {"d_c13store", "const float *"},
  {"d_c33store", "const float *"},
  {"d_c44store", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_minus_gravity_table", "const float *"},
  {"d_minus_deriv_gravity_table", "const float *"},
  {"d_density_table", "const float *"},
  {"wgll_cube", "const float *"},
  {"NSPEC_INNER_CORE_STRAIN_ONLY", "const int"},
  {"NSPEC_INNER_CORE", "const int"}
};

struct param_info_s noise_add_source_master_rec_kernel_params[] = {
  {"ibool", "const int *"},
  {"ispec_selected_rec", "const int *"},
  {"irec_master_noise", "const int"},
  {"accel", "float *"},
  {"noise_sourcearray", "const float *"},
  {"it", "const int"}
};

struct param_info_s noise_add_surface_movie_kernel_params[] = {
  {"accel", "float *"},
  {"ibool", "const int *"},
  {"ibelm_top", "const int *"},
  {"nspec_top", "const int"},
  {"noise_surface_movie", "const float *"},
  {"normal_x_noise", "const float *"},
  {"normal_y_noise", "const float *"},
  {"normal_z_noise", "const float *"},
  {"mask_noise", "const float *"},
  {"jacobian2D", "const float *"},
  {"wgllwgll", "const float *"}
};

struct param_info_s noise_transfer_surface_to_host_kernel_params[] = {
  {"ibelm_top", "const int *"},
  {"nspec_top", "const int"},
  {"ibool", "const int *"},
  {"displ", "const float *"},
  {"noise_surface_movie", "float *"}
};

struct param_info_s outer_core_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_potential", "const float *"},
  {"d_potential_dot_dot", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"wgllwgll_xz", "const float *"},
  {"wgllwgll_yz", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_d_ln_density_dr_table", "const float *"},
  {"d_minus_rho_g_over_kappa_fluid", "const float *"},
  {"wgll_cube", "const float *"},
  {"ROTATION", "const int"},
  {"time", "const float"},
  {"two_omega_earth", "const float"},
  {"deltat", "const float"},
  {"d_A_array_rotation", "float *"},
  {"d_B_array_rotation", "float *"},
  {"NSPEC_OUTER_CORE", "const int"}
};

struct param_info_s outer_core_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int"},
  {"d_ibool", "const int *"},
  {"d_phase_ispec_inner", "const int *"},
  {"num_phase_ispec", "const int"},
  {"d_iphase", "const int"},
  {"use_mesh_coloring_gpu", "const int"},
  {"d_potential", "const float *"},
  {"d_potential_dot_dot", "float *"},
  {"d_xix", "const float *"},
  {"d_xiy", "const float *"},
  {"d_xiz", "const float *"},
  {"d_etax", "const float *"},
  {"d_etay", "const float *"},
  {"d_etaz", "const float *"},
  {"d_gammax", "const float *"},
  {"d_gammay", "const float *"},
  {"d_gammaz", "const float *"},
  {"d_hprime_xx", "const float *"},
  {"d_hprimewgll_xx", "const float *"},
  {"wgllwgll_xy", "const float *"},
  {"wgllwgll_xz", "const float *"},
  {"wgllwgll_yz", "const float *"},
  {"GRAVITY", "const int"},
  {"d_xstore", "const float *"},
  {"d_ystore", "const float *"},
  {"d_zstore", "const float *"},
  {"d_d_ln_density_dr_table", "const float *"},
  {"d_minus_rho_g_over_kappa_fluid", "const float *"},
  {"wgll_cube", "const float *"},
  {"ROTATION", "const int"},
  {"time", "const float"},
  {"two_omega_earth", "const float"},
  {"deltat", "const float"},
  {"d_A_array_rotation", "float *"},
  {"d_B_array_rotation", "float *"},
  {"NSPEC_OUTER_CORE", "const int"}
};

struct param_info_s prepare_boundary_accel_on_device_params[] = {
  {"d_accel", "const float *"},
  {"d_send_accel_buffer", "float *"},
  {"num_interfaces", "const int"},
  {"max_nibool_interfaces", "const int"},
  {"d_nibool_interfaces", "const int *"},
  {"d_ibool_interfaces", "const int *"}
};

struct param_info_s prepare_boundary_potential_on_device_params[] = {
  {"d_potential_dot_dot_acoustic", "const float *"},
  {"d_send_potential_dot_dot_buffer", "float *"},
  {"num_interfaces", "const int"},
  {"max_nibool_interfaces", "const int"},
  {"d_nibool_interfaces", "const int *"},
  {"d_ibool_interfaces", "const int *"}
};

struct param_info_s update_accel_acoustic_kernel_params[] = {
  {"accel", "float *"},
  {"size", "const int"},
  {"rmass", "const float *"}
};

struct param_info_s update_accel_elastic_kernel_params[] = {
  {"accel", "float *"},
  {"veloc", "const float *"},
  {"size", "const int"},
  {"two_omega_earth", "const float"},
  {"rmassx", "const float *"},
  {"rmassy", "const float *"},
  {"rmassz", "const float *"}
};

struct param_info_s update_disp_veloc_kernel_params[] = {
  {"displ", "float *"},
  {"veloc", "float *"},
  {"accel", "float *"},
  {"size", "const int"},
  {"deltat", "const float"},
  {"deltatsqover2", "const float"},
  {"deltatover2", "const float"}
};

struct param_info_s update_potential_kernel_params[] = {
  {"potential_acoustic", "float *"},
  {"potential_dot_acoustic", "float *"},
  {"potential_dot_dot_acoustic", "float *"},
  {"size", "const int"},
  {"deltat", "const float"},
  {"deltatsqover2", "const float"},
  {"deltatover2", "const float"}
};

struct param_info_s update_veloc_acoustic_kernel_params[] = {
  {"veloc", "float *"},
  {"accel", "const float *"},
  {"size", "const int"},
  {"deltatover2", "const float"}
};

struct param_info_s update_veloc_elastic_kernel_params[] = {
  {"veloc", "float *"},
  {"accel", "const float *"},
  {"size", "const int"},
  {"deltatover2", "const float"}
};

struct param_info_s write_seismograms_transfer_from_device_kernel_params[] = {
  {"number_receiver_global", "const int *"},
  {"ispec_selected_rec", "const int *"},
  {"ibool", "const int *"},
  {"station_seismo_field", "float *"},
  {"d_field", "const float *"},
  {"nrec_local", "const int"}
};

struct param_info_s write_seismograms_transfer_strain_from_device_kernel_params[] = {
  {"number_receiver_global", "const int *"},
  {"ispec_selected_rec", "const int *"},
  {"ibool", "const int *"},
  {"station_strain_field", "float *"},
  {"d_field", "const float *"},
  {"nrec_local", "const int"}
};

static struct kernel_lookup_s function_lookup_table[] = { 
  {(void *) 0x4e3619, "assemble_boundary_accel_on_device", 6, assemble_boundary_accel_on_device_params},
  {(void *) 0x4e3839, "assemble_boundary_potential_on_device", 6, assemble_boundary_potential_on_device_params},
  {(void *) 0x4eb9ec, "compute_acoustic_kernel", 20, compute_acoustic_kernel_params},
  {(void *) 0x4e41d5, "compute_add_sources_adjoint_kernel", 7, compute_add_sources_adjoint_kernel_params},
  {(void *) 0x4e4440, "compute_add_sources_kernel", 8, compute_add_sources_kernel_params},
  {(void *) 0x4eb306, "compute_ani_kernel", 15, compute_ani_kernel_params},
  {(void *) 0x4eea15, "compute_ani_undo_att_kernel", 21, compute_ani_undo_att_kernel_params},
  {(void *) 0x4e4d5a, "compute_coupling_CMB_fluid_kernel", 14, compute_coupling_CMB_fluid_kernel_params},
  {(void *) 0x4e510a, "compute_coupling_ICB_fluid_kernel", 14, compute_coupling_ICB_fluid_kernel_params},
  {(void *) 0x4e4708, "compute_coupling_fluid_CMB_kernel", 10, compute_coupling_fluid_CMB_kernel_params},
  {(void *) 0x4e49e0, "compute_coupling_fluid_ICB_kernel", 10, compute_coupling_fluid_ICB_kernel_params},
  {(void *) 0x4e53c8, "compute_coupling_ocean_kernel", 8, compute_coupling_ocean_kernel_params},
  {(void *) 0x4eb57c, "compute_hess_kernel", 6, compute_hess_kernel_params},
  {(void *) 0x4eaef9, "compute_iso_kernel", 16, compute_iso_kernel_params},
  {(void *) 0x4eef5f, "compute_iso_undo_att_kernel", 22, compute_iso_undo_att_kernel_params},
  {(void *) 0x4eab54, "compute_rho_kernel", 6, compute_rho_kernel_params},
  {(void *) 0x4e6656, "compute_stacey_acoustic_backward_kernel", 12, compute_stacey_acoustic_backward_kernel_params},
  {(void *) 0x4e62f4, "compute_stacey_acoustic_kernel", 17, compute_stacey_acoustic_kernel_params},
  {(void *) 0x4e6e0a, "compute_stacey_elastic_backward_kernel", 12, compute_stacey_elastic_backward_kernel_params},
  {(void *) 0x4e6a98, "compute_stacey_elastic_kernel", 19, compute_stacey_elastic_kernel_params},
  {(void *) 0x4ebd31, "compute_strength_noise_kernel", 10, compute_strength_noise_kernel_params},
  {(void *) 0x4ee156, "crust_mantle_impl_kernel_adjoint", 80, crust_mantle_impl_kernel_adjoint_params},
  {(void *) 0x4ecd5a, "crust_mantle_impl_kernel_forward", 80, crust_mantle_impl_kernel_forward_params},
  {(void *) 0x4e3e20, "get_maximum_scalar_kernel", 3, get_maximum_scalar_kernel_params},
  {(void *) 0x4e3fac, "get_maximum_vector_kernel", 3, get_maximum_vector_kernel_params},
  {(void *) 0x4ea647, "inner_core_impl_kernel_adjoint", 62, inner_core_impl_kernel_adjoint_params},
  {(void *) 0x4e977b, "inner_core_impl_kernel_forward", 62, inner_core_impl_kernel_forward_params},
  {(void *) 0x4e5c2a, "noise_add_source_master_rec_kernel", 6, noise_add_source_master_rec_kernel_params},
  {(void *) 0x4e5f09, "noise_add_surface_movie_kernel", 11, noise_add_surface_movie_kernel_params},
  {(void *) 0x4e5a14, "noise_transfer_surface_to_host_kernel", 5, noise_transfer_surface_to_host_kernel_params},
  {(void *) 0x4e8a24, "outer_core_impl_kernel_adjoint", 36, outer_core_impl_kernel_adjoint_params},
  {(void *) 0x4e8190, "outer_core_impl_kernel_forward", 36, outer_core_impl_kernel_forward_params},
  {(void *) 0x4e3c79, "prepare_boundary_accel_on_device", 6, prepare_boundary_accel_on_device_params},
  {(void *) 0x4e3a59, "prepare_boundary_potential_on_device", 6, prepare_boundary_potential_on_device_params},
  {(void *) 0x4e78cc, "update_accel_acoustic_kernel", 3, update_accel_acoustic_kernel_params},
  {(void *) 0x4e755d, "update_accel_elastic_kernel", 7, update_accel_elastic_kernel_params},
  {(void *) 0x4e7087, "update_disp_veloc_kernel", 7, update_disp_veloc_kernel_params},
  {(void *) 0x4e72f3, "update_potential_kernel", 7, update_potential_kernel_params},
  {(void *) 0x4e7a7f, "update_veloc_acoustic_kernel", 4, update_veloc_acoustic_kernel_params},
  {(void *) 0x4e7733, "update_veloc_elastic_kernel", 4, update_veloc_elastic_kernel_params},
  {(void *) 0x4e55f7, "write_seismograms_transfer_from_device_kernel", 6, write_seismograms_transfer_from_device_kernel_params},
  {(void *) 0x4e581b, "write_seismograms_transfer_strain_from_device_kernel", 6, write_seismograms_transfer_strain_from_device_kernel_params}
};

struct kernel_lookup_s *cuda_get_lookup_table(void) {
    return function_lookup_table;
}
