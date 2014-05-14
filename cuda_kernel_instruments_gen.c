#include <stddef.h>
#include <stdio.h>

#define NB_CUDA_KERNEL 42

struct param_info_s {
  const char *name;
  const char *type;
  size_t offset;
};

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
  {"d_accel", "float *", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->d_accel},
  {"d_send_accel_buffer", "const float *", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->d_send_accel_buffer},
  {"num_interfaces", "const int", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->num_interfaces},
  {"max_nibool_interfaces", "const int", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->max_nibool_interfaces},
  {"d_nibool_interfaces", "const int *", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->d_nibool_interfaces},
  {"d_ibool_interfaces", "const int *", (size_t) &((struct assemble_boundary_accel_on_device_args_s *) NULL)->d_ibool_interfaces}
};

struct param_info_s assemble_boundary_potential_on_device_params[] = {
  {"d_potential_dot_dot_acoustic", "float *", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->d_potential_dot_dot_acoustic},
  {"d_send_potential_dot_dot_buffer", "const float *", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->d_send_potential_dot_dot_buffer},
  {"num_interfaces", "const int", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->num_interfaces},
  {"max_nibool_interfaces", "const int", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->max_nibool_interfaces},
  {"d_nibool_interfaces", "const int *", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->d_nibool_interfaces},
  {"d_ibool_interfaces", "const int *", (size_t) &((struct assemble_boundary_potential_on_device_args_s *) NULL)->d_ibool_interfaces}
};

struct param_info_s compute_acoustic_kernel_params[] = {
  {"ibool", "const int *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->ibool},
  {"rhostore", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->rhostore},
  {"kappastore", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->kappastore},
  {"hprime_xx", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->hprime_xx},
  {"d_xix", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->d_gammaz},
  {"potential_dot_dot_acoustic", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->potential_dot_dot_acoustic},
  {"b_potential_acoustic", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->b_potential_acoustic},
  {"b_potential_dot_dot_acoustic", "const float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->b_potential_dot_dot_acoustic},
  {"rho_ac_kl", "float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->rho_ac_kl},
  {"kappa_ac_kl", "float *", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->kappa_ac_kl},
  {"deltat", "const float", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->deltat},
  {"NSPEC", "const int", (size_t) &((struct compute_acoustic_kernel_args_s *) NULL)->NSPEC}
};

struct param_info_s compute_add_sources_adjoint_kernel_params[] = {
  {"accel", "float *", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->accel},
  {"nrec", "const int", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->nrec},
  {"adj_sourcearrays", "const float *", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->adj_sourcearrays},
  {"ibool", "const int *", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->ibool},
  {"ispec_selected_rec", "const int *", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->ispec_selected_rec},
  {"pre_computed_irec", "const int *", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->pre_computed_irec},
  {"nadj_rec_local", "const int", (size_t) &((struct compute_add_sources_adjoint_kernel_args_s *) NULL)->nadj_rec_local}
};

struct param_info_s compute_add_sources_kernel_params[] = {
  {"accel", "float *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->accel},
  {"ibool", "const int *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->ibool},
  {"sourcearrays", "const float *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->sourcearrays},
  {"stf_pre_compute", "const double *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->stf_pre_compute},
  {"myrank", "const int", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->myrank},
  {"islice_selected_source", "const int *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->islice_selected_source},
  {"ispec_selected_source", "const int *", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->ispec_selected_source},
  {"NSOURCES", "const int", (size_t) &((struct compute_add_sources_kernel_args_s *) NULL)->NSOURCES}
};

struct param_info_s compute_ani_kernel_params[] = {
  {"epsilondev_xx", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->epsilon_trace_over_3},
  {"b_epsilondev_xx", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilondev_xx},
  {"b_epsilondev_yy", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilondev_yy},
  {"b_epsilondev_xy", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilondev_xy},
  {"b_epsilondev_xz", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilondev_xz},
  {"b_epsilondev_yz", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilondev_yz},
  {"b_epsilon_trace_over_3", "const float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->b_epsilon_trace_over_3},
  {"cijkl_kl", "float *", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->cijkl_kl},
  {"NSPEC", "const int", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->NSPEC},
  {"deltat", "const float", (size_t) &((struct compute_ani_kernel_args_s *) NULL)->deltat}
};

struct param_info_s compute_ani_undo_att_kernel_params[] = {
  {"epsilondev_xx", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->epsilon_trace_over_3},
  {"cijkl_kl", "float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->cijkl_kl},
  {"NSPEC", "const int", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->NSPEC},
  {"deltat", "const float", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->deltat},
  {"d_ibool", "const int *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_ibool},
  {"d_b_displ", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_b_displ},
  {"d_xix", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct compute_ani_undo_att_kernel_args_s *) NULL)->d_hprime_xx}
};

struct param_info_s compute_coupling_CMB_fluid_kernel_params[] = {
  {"displ_crust_mantle", "const float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->displ_crust_mantle},
  {"accel_crust_mantle", "float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->accel_crust_mantle},
  {"accel_outer_core", "const float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->accel_outer_core},
  {"ibool_crust_mantle", "const int *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->ibool_crust_mantle},
  {"ibelm_bottom_crust_mantle", "const int *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->ibelm_bottom_crust_mantle},
  {"normal_top_outer_core", "const float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->normal_top_outer_core},
  {"jacobian2D_top_outer_core", "const float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->jacobian2D_top_outer_core},
  {"wgllwgll_xy", "const float *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->wgllwgll_xy},
  {"ibool_outer_core", "const int *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->ibool_outer_core},
  {"ibelm_top_outer_core", "const int *", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->ibelm_top_outer_core},
  {"RHO_TOP_OC", "const float", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->RHO_TOP_OC},
  {"minus_g_cmb", "const float", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->minus_g_cmb},
  {"GRAVITY", "int", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->GRAVITY},
  {"NSPEC2D_BOTTOM_CM", "const int", (size_t) &((struct compute_coupling_CMB_fluid_kernel_args_s *) NULL)->NSPEC2D_BOTTOM_CM}
};

struct param_info_s compute_coupling_ICB_fluid_kernel_params[] = {
  {"displ_inner_core", "const float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->displ_inner_core},
  {"accel_inner_core", "float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->accel_inner_core},
  {"accel_outer_core", "const float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->accel_outer_core},
  {"ibool_inner_core", "const int *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->ibool_inner_core},
  {"ibelm_top_inner_core", "const int *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->ibelm_top_inner_core},
  {"normal_bottom_outer_core", "const float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->normal_bottom_outer_core},
  {"jacobian2D_bottom_outer_core", "const float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->jacobian2D_bottom_outer_core},
  {"wgllwgll_xy", "const float *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->wgllwgll_xy},
  {"ibool_outer_core", "const int *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->ibool_outer_core},
  {"ibelm_bottom_outer_core", "const int *", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->ibelm_bottom_outer_core},
  {"RHO_BOTTOM_OC", "const float", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->RHO_BOTTOM_OC},
  {"minus_g_icb", "const float", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->minus_g_icb},
  {"GRAVITY", "int", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->GRAVITY},
  {"NSPEC2D_TOP_IC", "const int", (size_t) &((struct compute_coupling_ICB_fluid_kernel_args_s *) NULL)->NSPEC2D_TOP_IC}
};

struct param_info_s compute_coupling_fluid_CMB_kernel_params[] = {
  {"displ_crust_mantle", "const float *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->displ_crust_mantle},
  {"accel_outer_core", "float *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->accel_outer_core},
  {"ibool_crust_mantle", "const int *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->ibool_crust_mantle},
  {"ibelm_bottom_crust_mantle", "const int *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->ibelm_bottom_crust_mantle},
  {"normal_top_outer_core", "const float *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->normal_top_outer_core},
  {"jacobian2D_top_outer_core", "const float *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->jacobian2D_top_outer_core},
  {"wgllwgll_xy", "const float *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->wgllwgll_xy},
  {"ibool_outer_core", "const int *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->ibool_outer_core},
  {"ibelm_top_outer_core", "const int *", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->ibelm_top_outer_core},
  {"NSPEC2D_TOP_OC", "const int", (size_t) &((struct compute_coupling_fluid_CMB_kernel_args_s *) NULL)->NSPEC2D_TOP_OC}
};

struct param_info_s compute_coupling_fluid_ICB_kernel_params[] = {
  {"displ_inner_core", "const float *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->displ_inner_core},
  {"accel_outer_core", "float *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->accel_outer_core},
  {"ibool_inner_core", "const int *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->ibool_inner_core},
  {"ibelm_top_inner_core", "const int *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->ibelm_top_inner_core},
  {"normal_bottom_outer_core", "const float *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->normal_bottom_outer_core},
  {"jacobian2D_bottom_outer_core", "const float *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->jacobian2D_bottom_outer_core},
  {"wgllwgll_xy", "const float *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->wgllwgll_xy},
  {"ibool_outer_core", "const int *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->ibool_outer_core},
  {"ibelm_bottom_outer_core", "const int *", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->ibelm_bottom_outer_core},
  {"NSPEC2D_BOTTOM_OC", "const int", (size_t) &((struct compute_coupling_fluid_ICB_kernel_args_s *) NULL)->NSPEC2D_BOTTOM_OC}
};

struct param_info_s compute_coupling_ocean_kernel_params[] = {
  {"accel_crust_mantle", "float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->accel_crust_mantle},
  {"rmassx_crust_mantle", "const float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->rmassx_crust_mantle},
  {"rmassy_crust_mantle", "const float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->rmassy_crust_mantle},
  {"rmassz_crust_mantle", "const float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->rmassz_crust_mantle},
  {"rmass_ocean_load", "const float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->rmass_ocean_load},
  {"npoin_ocean_load", "const int", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->npoin_ocean_load},
  {"ibool_ocean_load", "const int *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->ibool_ocean_load},
  {"normal_ocean_load", "const float *", (size_t) &((struct compute_coupling_ocean_kernel_args_s *) NULL)->normal_ocean_load}
};

struct param_info_s compute_hess_kernel_params[] = {
  {"ibool", "const int *", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->ibool},
  {"accel", "const float *", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->accel},
  {"b_accel", "const float *", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->b_accel},
  {"hess_kl", "float *", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->hess_kl},
  {"deltat", "const float", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->deltat},
  {"NSPEC_AB", "const int", (size_t) &((struct compute_hess_kernel_args_s *) NULL)->NSPEC_AB}
};

struct param_info_s compute_iso_kernel_params[] = {
  {"epsilondev_xx", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->epsilon_trace_over_3},
  {"b_epsilondev_xx", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilondev_xx},
  {"b_epsilondev_yy", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilondev_yy},
  {"b_epsilondev_xy", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilondev_xy},
  {"b_epsilondev_xz", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilondev_xz},
  {"b_epsilondev_yz", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilondev_yz},
  {"b_epsilon_trace_over_3", "const float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->b_epsilon_trace_over_3},
  {"mu_kl", "float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->mu_kl},
  {"kappa_kl", "float *", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->kappa_kl},
  {"NSPEC", "const int", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->NSPEC},
  {"deltat", "const float", (size_t) &((struct compute_iso_kernel_args_s *) NULL)->deltat}
};

struct param_info_s compute_iso_undo_att_kernel_params[] = {
  {"epsilondev_xx", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->epsilon_trace_over_3},
  {"mu_kl", "float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->mu_kl},
  {"kappa_kl", "float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->kappa_kl},
  {"NSPEC", "const int", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->NSPEC},
  {"deltat", "const float", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->deltat},
  {"d_ibool", "const int *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_ibool},
  {"d_b_displ", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_b_displ},
  {"d_xix", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct compute_iso_undo_att_kernel_args_s *) NULL)->d_hprime_xx}
};

struct param_info_s compute_rho_kernel_params[] = {
  {"ibool", "const int *", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->ibool},
  {"accel", "const float *", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->accel},
  {"b_displ", "const float *", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->b_displ},
  {"rho_kl", "float *", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->rho_kl},
  {"NSPEC", "const int", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->NSPEC},
  {"deltat", "const float", (size_t) &((struct compute_rho_kernel_args_s *) NULL)->deltat}
};

struct param_info_s compute_stacey_acoustic_backward_kernel_params[] = {
  {"b_potential_dot_dot_acoustic", "float *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->b_potential_dot_dot_acoustic},
  {"b_absorb_potential", "const float *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->b_absorb_potential},
  {"interface_type", "const int", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->interface_type},
  {"num_abs_boundary_faces", "const int", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->num_abs_boundary_faces},
  {"abs_boundary_ispec", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->abs_boundary_ispec},
  {"nkmin_xi", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->nkmin_xi},
  {"nkmin_eta", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->nkmin_eta},
  {"njmin", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->njmin},
  {"njmax", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->njmax},
  {"nimin", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->nimin},
  {"nimax", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->nimax},
  {"ibool", "const int *", (size_t) &((struct compute_stacey_acoustic_backward_kernel_args_s *) NULL)->ibool}
};

struct param_info_s compute_stacey_acoustic_kernel_params[] = {
  {"potential_dot_acoustic", "const float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->potential_dot_acoustic},
  {"potential_dot_dot_acoustic", "float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->potential_dot_dot_acoustic},
  {"interface_type", "const int", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->interface_type},
  {"num_abs_boundary_faces", "const int", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->num_abs_boundary_faces},
  {"abs_boundary_ispec", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->abs_boundary_ispec},
  {"nkmin_xi", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->nkmin_xi},
  {"nkmin_eta", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->nkmin_eta},
  {"njmin", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->njmin},
  {"njmax", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->njmax},
  {"nimin", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->nimin},
  {"nimax", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->nimax},
  {"abs_boundary_jacobian2D", "const float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->abs_boundary_jacobian2D},
  {"wgllwgll", "const float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->wgllwgll},
  {"ibool", "const int *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->ibool},
  {"vpstore", "const float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->vpstore},
  {"SAVE_FORWARD", "const int", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->SAVE_FORWARD},
  {"b_absorb_potential", "float *", (size_t) &((struct compute_stacey_acoustic_kernel_args_s *) NULL)->b_absorb_potential}
};

struct param_info_s compute_stacey_elastic_backward_kernel_params[] = {
  {"b_accel", "float *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->b_accel},
  {"b_absorb_field", "const float *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->b_absorb_field},
  {"interface_type", "const int", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->interface_type},
  {"num_abs_boundary_faces", "const int", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->num_abs_boundary_faces},
  {"abs_boundary_ispec", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->abs_boundary_ispec},
  {"nkmin_xi", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->nkmin_xi},
  {"nkmin_eta", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->nkmin_eta},
  {"njmin", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->njmin},
  {"njmax", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->njmax},
  {"nimin", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->nimin},
  {"nimax", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->nimax},
  {"ibool", "const int *", (size_t) &((struct compute_stacey_elastic_backward_kernel_args_s *) NULL)->ibool}
};

struct param_info_s compute_stacey_elastic_kernel_params[] = {
  {"veloc", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->veloc},
  {"accel", "float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->accel},
  {"interface_type", "const int", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->interface_type},
  {"num_abs_boundary_faces", "const int", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->num_abs_boundary_faces},
  {"abs_boundary_ispec", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->abs_boundary_ispec},
  {"nkmin_xi", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->nkmin_xi},
  {"nkmin_eta", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->nkmin_eta},
  {"njmin", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->njmin},
  {"njmax", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->njmax},
  {"nimin", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->nimin},
  {"nimax", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->nimax},
  {"abs_boundary_normal", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->abs_boundary_normal},
  {"abs_boundary_jacobian2D", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->abs_boundary_jacobian2D},
  {"wgllwgll", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->wgllwgll},
  {"ibool", "const int *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->ibool},
  {"rho_vp", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->rho_vp},
  {"rho_vs", "const float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->rho_vs},
  {"SAVE_FORWARD", "const int", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->SAVE_FORWARD},
  {"b_absorb_field", "float *", (size_t) &((struct compute_stacey_elastic_kernel_args_s *) NULL)->b_absorb_field}
};

struct param_info_s compute_strength_noise_kernel_params[] = {
  {"displ", "const float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->displ},
  {"ibelm_top", "const int *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->ibelm_top},
  {"ibool", "const int *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->ibool},
  {"noise_surface_movie", "const float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->noise_surface_movie},
  {"normal_x_noise", "const float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->normal_x_noise},
  {"normal_y_noise", "const float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->normal_y_noise},
  {"normal_z_noise", "const float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->normal_z_noise},
  {"Sigma_kl", "float *", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->Sigma_kl},
  {"deltat", "const float", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->deltat},
  {"nspec_top", "const int", (size_t) &((struct compute_strength_noise_kernel_args_s *) NULL)->nspec_top}
};

struct param_info_s crust_mantle_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_ibool},
  {"d_ispec_is_tiso", "const int *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_ispec_is_tiso},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_iphase},
  {"deltat", "const float", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->deltat},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_displ", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_displ},
  {"d_accel", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_accel},
  {"d_xix", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_hprimewgll_xx},
  {"d_wgllwgll_xy", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_xy},
  {"d_wgllwgll_xz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_xz},
  {"d_wgllwgll_yz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_yz},
  {"d_kappavstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_kappavstore},
  {"d_muvstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_muvstore},
  {"d_kappahstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_kappahstore},
  {"d_muhstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_muhstore},
  {"d_eta_anisostore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_eta_anisostore},
  {"COMPUTE_AND_STORE_STRAIN", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->COMPUTE_AND_STORE_STRAIN},
  {"epsilondev_xx", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->epsilon_trace_over_3},
  {"ATTENUATION", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->ATTENUATION},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->PARTIAL_PHYS_DISPERSION_ONLY},
  {"USE_3D_ATTENUATION_ARRAYS", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->USE_3D_ATTENUATION_ARRAYS},
  {"one_minus_sum_beta", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->one_minus_sum_beta},
  {"factor_common", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->factor_common},
  {"R_xx", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->R_xx},
  {"R_yy", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->R_yy},
  {"R_xy", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->R_xy},
  {"R_xz", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->R_xz},
  {"R_yz", "float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->R_yz},
  {"alphaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->alphaval},
  {"betaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->betaval},
  {"gammaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->gammaval},
  {"ANISOTROPY", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->ANISOTROPY},
  {"d_c11store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c11store},
  {"d_c12store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c12store},
  {"d_c13store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c13store},
  {"d_c14store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c14store},
  {"d_c15store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c15store},
  {"d_c16store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c16store},
  {"d_c22store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c22store},
  {"d_c23store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c23store},
  {"d_c24store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c24store},
  {"d_c25store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c25store},
  {"d_c26store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c26store},
  {"d_c33store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c33store},
  {"d_c34store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c34store},
  {"d_c35store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c35store},
  {"d_c36store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c36store},
  {"d_c44store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c44store},
  {"d_c45store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c45store},
  {"d_c46store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c46store},
  {"d_c55store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c55store},
  {"d_c56store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c56store},
  {"d_c66store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_c66store},
  {"GRAVITY", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_zstore},
  {"d_minus_gravity_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_minus_gravity_table},
  {"d_minus_deriv_gravity_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_minus_deriv_gravity_table},
  {"d_density_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->d_density_table},
  {"wgll_cube", "const float *", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->wgll_cube},
  {"NSPEC_CRUST_MANTLE_STRAIN_ONLY", "const int", (size_t) &((struct crust_mantle_impl_kernel_adjoint_args_s *) NULL)->NSPEC_CRUST_MANTLE_STRAIN_ONLY}
};

struct param_info_s crust_mantle_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_ibool},
  {"d_ispec_is_tiso", "const int *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_ispec_is_tiso},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_iphase},
  {"deltat", "const float", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->deltat},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_displ", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_displ},
  {"d_accel", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_accel},
  {"d_xix", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_hprimewgll_xx},
  {"d_wgllwgll_xy", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_xy},
  {"d_wgllwgll_xz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_xz},
  {"d_wgllwgll_yz", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_yz},
  {"d_kappavstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_kappavstore},
  {"d_muvstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_muvstore},
  {"d_kappahstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_kappahstore},
  {"d_muhstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_muhstore},
  {"d_eta_anisostore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_eta_anisostore},
  {"COMPUTE_AND_STORE_STRAIN", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->COMPUTE_AND_STORE_STRAIN},
  {"epsilondev_xx", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->epsilon_trace_over_3},
  {"ATTENUATION", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->ATTENUATION},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->PARTIAL_PHYS_DISPERSION_ONLY},
  {"USE_3D_ATTENUATION_ARRAYS", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->USE_3D_ATTENUATION_ARRAYS},
  {"one_minus_sum_beta", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->one_minus_sum_beta},
  {"factor_common", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->factor_common},
  {"R_xx", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->R_xx},
  {"R_yy", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->R_yy},
  {"R_xy", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->R_xy},
  {"R_xz", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->R_xz},
  {"R_yz", "float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->R_yz},
  {"alphaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->alphaval},
  {"betaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->betaval},
  {"gammaval", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->gammaval},
  {"ANISOTROPY", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->ANISOTROPY},
  {"d_c11store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c11store},
  {"d_c12store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c12store},
  {"d_c13store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c13store},
  {"d_c14store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c14store},
  {"d_c15store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c15store},
  {"d_c16store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c16store},
  {"d_c22store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c22store},
  {"d_c23store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c23store},
  {"d_c24store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c24store},
  {"d_c25store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c25store},
  {"d_c26store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c26store},
  {"d_c33store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c33store},
  {"d_c34store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c34store},
  {"d_c35store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c35store},
  {"d_c36store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c36store},
  {"d_c44store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c44store},
  {"d_c45store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c45store},
  {"d_c46store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c46store},
  {"d_c55store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c55store},
  {"d_c56store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c56store},
  {"d_c66store", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_c66store},
  {"GRAVITY", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_zstore},
  {"d_minus_gravity_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_minus_gravity_table},
  {"d_minus_deriv_gravity_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_minus_deriv_gravity_table},
  {"d_density_table", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->d_density_table},
  {"wgll_cube", "const float *", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->wgll_cube},
  {"NSPEC_CRUST_MANTLE_STRAIN_ONLY", "const int", (size_t) &((struct crust_mantle_impl_kernel_forward_args_s *) NULL)->NSPEC_CRUST_MANTLE_STRAIN_ONLY}
};

struct param_info_s get_maximum_scalar_kernel_params[] = {
  {"array", "const float *", (size_t) &((struct get_maximum_scalar_kernel_args_s *) NULL)->array},
  {"size", "const int", (size_t) &((struct get_maximum_scalar_kernel_args_s *) NULL)->size},
  {"d_max", "float *", (size_t) &((struct get_maximum_scalar_kernel_args_s *) NULL)->d_max}
};

struct param_info_s get_maximum_vector_kernel_params[] = {
  {"array", "const float *", (size_t) &((struct get_maximum_vector_kernel_args_s *) NULL)->array},
  {"size", "const int", (size_t) &((struct get_maximum_vector_kernel_args_s *) NULL)->size},
  {"d_max", "float *", (size_t) &((struct get_maximum_vector_kernel_args_s *) NULL)->d_max}
};

struct param_info_s inner_core_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_ibool},
  {"d_idoubling", "const int *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_idoubling},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_iphase},
  {"deltat", "const float", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->deltat},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_displ", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_displ},
  {"d_accel", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_accel},
  {"d_xix", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_hprimewgll_xx},
  {"d_wgllwgll_xy", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_xy},
  {"d_wgllwgll_xz", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_xz},
  {"d_wgllwgll_yz", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_wgllwgll_yz},
  {"d_kappavstore", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_kappavstore},
  {"d_muvstore", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_muvstore},
  {"COMPUTE_AND_STORE_STRAIN", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->COMPUTE_AND_STORE_STRAIN},
  {"epsilondev_xx", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->epsilon_trace_over_3},
  {"ATTENUATION", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->ATTENUATION},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->PARTIAL_PHYS_DISPERSION_ONLY},
  {"USE_3D_ATTENUATION_ARRAYS", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->USE_3D_ATTENUATION_ARRAYS},
  {"one_minus_sum_beta", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->one_minus_sum_beta},
  {"factor_common", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->factor_common},
  {"R_xx", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->R_xx},
  {"R_yy", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->R_yy},
  {"R_xy", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->R_xy},
  {"R_xz", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->R_xz},
  {"R_yz", "float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->R_yz},
  {"alphaval", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->alphaval},
  {"betaval", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->betaval},
  {"gammaval", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->gammaval},
  {"ANISOTROPY", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->ANISOTROPY},
  {"d_c11store", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_c11store},
  {"d_c12store", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_c12store},
  {"d_c13store", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_c13store},
  {"d_c33store", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_c33store},
  {"d_c44store", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_c44store},
  {"GRAVITY", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_zstore},
  {"d_minus_gravity_table", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_minus_gravity_table},
  {"d_minus_deriv_gravity_table", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_minus_deriv_gravity_table},
  {"d_density_table", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->d_density_table},
  {"wgll_cube", "const float *", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->wgll_cube},
  {"NSPEC_INNER_CORE_STRAIN_ONLY", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->NSPEC_INNER_CORE_STRAIN_ONLY},
  {"NSPEC_INNER_CORE", "const int", (size_t) &((struct inner_core_impl_kernel_adjoint_args_s *) NULL)->NSPEC_INNER_CORE}
};

struct param_info_s inner_core_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_ibool},
  {"d_idoubling", "const int *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_idoubling},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_iphase},
  {"deltat", "const float", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->deltat},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_displ", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_displ},
  {"d_accel", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_accel},
  {"d_xix", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_hprimewgll_xx},
  {"d_wgllwgll_xy", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_xy},
  {"d_wgllwgll_xz", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_xz},
  {"d_wgllwgll_yz", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_wgllwgll_yz},
  {"d_kappavstore", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_kappavstore},
  {"d_muvstore", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_muvstore},
  {"COMPUTE_AND_STORE_STRAIN", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->COMPUTE_AND_STORE_STRAIN},
  {"epsilondev_xx", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilondev_xx},
  {"epsilondev_yy", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilondev_yy},
  {"epsilondev_xy", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilondev_xy},
  {"epsilondev_xz", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilondev_xz},
  {"epsilondev_yz", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilondev_yz},
  {"epsilon_trace_over_3", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->epsilon_trace_over_3},
  {"ATTENUATION", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->ATTENUATION},
  {"PARTIAL_PHYS_DISPERSION_ONLY", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->PARTIAL_PHYS_DISPERSION_ONLY},
  {"USE_3D_ATTENUATION_ARRAYS", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->USE_3D_ATTENUATION_ARRAYS},
  {"one_minus_sum_beta", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->one_minus_sum_beta},
  {"factor_common", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->factor_common},
  {"R_xx", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->R_xx},
  {"R_yy", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->R_yy},
  {"R_xy", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->R_xy},
  {"R_xz", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->R_xz},
  {"R_yz", "float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->R_yz},
  {"alphaval", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->alphaval},
  {"betaval", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->betaval},
  {"gammaval", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->gammaval},
  {"ANISOTROPY", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->ANISOTROPY},
  {"d_c11store", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_c11store},
  {"d_c12store", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_c12store},
  {"d_c13store", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_c13store},
  {"d_c33store", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_c33store},
  {"d_c44store", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_c44store},
  {"GRAVITY", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_zstore},
  {"d_minus_gravity_table", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_minus_gravity_table},
  {"d_minus_deriv_gravity_table", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_minus_deriv_gravity_table},
  {"d_density_table", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->d_density_table},
  {"wgll_cube", "const float *", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->wgll_cube},
  {"NSPEC_INNER_CORE_STRAIN_ONLY", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->NSPEC_INNER_CORE_STRAIN_ONLY},
  {"NSPEC_INNER_CORE", "const int", (size_t) &((struct inner_core_impl_kernel_forward_args_s *) NULL)->NSPEC_INNER_CORE}
};

struct param_info_s noise_add_source_master_rec_kernel_params[] = {
  {"ibool", "const int *", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->ibool},
  {"ispec_selected_rec", "const int *", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->ispec_selected_rec},
  {"irec_master_noise", "const int", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->irec_master_noise},
  {"accel", "float *", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->accel},
  {"noise_sourcearray", "const float *", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->noise_sourcearray},
  {"it", "const int", (size_t) &((struct noise_add_source_master_rec_kernel_args_s *) NULL)->it}
};

struct param_info_s noise_add_surface_movie_kernel_params[] = {
  {"accel", "float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->accel},
  {"ibool", "const int *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->ibool},
  {"ibelm_top", "const int *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->ibelm_top},
  {"nspec_top", "const int", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->nspec_top},
  {"noise_surface_movie", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->noise_surface_movie},
  {"normal_x_noise", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->normal_x_noise},
  {"normal_y_noise", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->normal_y_noise},
  {"normal_z_noise", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->normal_z_noise},
  {"mask_noise", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->mask_noise},
  {"jacobian2D", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->jacobian2D},
  {"wgllwgll", "const float *", (size_t) &((struct noise_add_surface_movie_kernel_args_s *) NULL)->wgllwgll}
};

struct param_info_s noise_transfer_surface_to_host_kernel_params[] = {
  {"ibelm_top", "const int *", (size_t) &((struct noise_transfer_surface_to_host_kernel_args_s *) NULL)->ibelm_top},
  {"nspec_top", "const int", (size_t) &((struct noise_transfer_surface_to_host_kernel_args_s *) NULL)->nspec_top},
  {"ibool", "const int *", (size_t) &((struct noise_transfer_surface_to_host_kernel_args_s *) NULL)->ibool},
  {"displ", "const float *", (size_t) &((struct noise_transfer_surface_to_host_kernel_args_s *) NULL)->displ},
  {"noise_surface_movie", "float *", (size_t) &((struct noise_transfer_surface_to_host_kernel_args_s *) NULL)->noise_surface_movie}
};

struct param_info_s outer_core_impl_kernel_adjoint_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_ibool},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_iphase},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_potential", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_potential},
  {"d_potential_dot_dot", "float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_potential_dot_dot},
  {"d_xix", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_hprimewgll_xx},
  {"wgllwgll_xy", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->wgllwgll_xy},
  {"wgllwgll_xz", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->wgllwgll_xz},
  {"wgllwgll_yz", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->wgllwgll_yz},
  {"GRAVITY", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_zstore},
  {"d_d_ln_density_dr_table", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_d_ln_density_dr_table},
  {"d_minus_rho_g_over_kappa_fluid", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_minus_rho_g_over_kappa_fluid},
  {"wgll_cube", "const float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->wgll_cube},
  {"ROTATION", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->ROTATION},
  {"time", "const float", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->time},
  {"two_omega_earth", "const float", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->two_omega_earth},
  {"deltat", "const float", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->deltat},
  {"d_A_array_rotation", "float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_A_array_rotation},
  {"d_B_array_rotation", "float *", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->d_B_array_rotation},
  {"NSPEC_OUTER_CORE", "const int", (size_t) &((struct outer_core_impl_kernel_adjoint_args_s *) NULL)->NSPEC_OUTER_CORE}
};

struct param_info_s outer_core_impl_kernel_forward_params[] = {
  {"nb_blocks_to_compute", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->nb_blocks_to_compute},
  {"d_ibool", "const int *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_ibool},
  {"d_phase_ispec_inner", "const int *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_phase_ispec_inner},
  {"num_phase_ispec", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->num_phase_ispec},
  {"d_iphase", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_iphase},
  {"use_mesh_coloring_gpu", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->use_mesh_coloring_gpu},
  {"d_potential", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_potential},
  {"d_potential_dot_dot", "float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_potential_dot_dot},
  {"d_xix", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_xix},
  {"d_xiy", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_xiy},
  {"d_xiz", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_xiz},
  {"d_etax", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_etax},
  {"d_etay", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_etay},
  {"d_etaz", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_etaz},
  {"d_gammax", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_gammax},
  {"d_gammay", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_gammay},
  {"d_gammaz", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_gammaz},
  {"d_hprime_xx", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_hprime_xx},
  {"d_hprimewgll_xx", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_hprimewgll_xx},
  {"wgllwgll_xy", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->wgllwgll_xy},
  {"wgllwgll_xz", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->wgllwgll_xz},
  {"wgllwgll_yz", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->wgllwgll_yz},
  {"GRAVITY", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->GRAVITY},
  {"d_xstore", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_xstore},
  {"d_ystore", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_ystore},
  {"d_zstore", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_zstore},
  {"d_d_ln_density_dr_table", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_d_ln_density_dr_table},
  {"d_minus_rho_g_over_kappa_fluid", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_minus_rho_g_over_kappa_fluid},
  {"wgll_cube", "const float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->wgll_cube},
  {"ROTATION", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->ROTATION},
  {"time", "const float", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->time},
  {"two_omega_earth", "const float", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->two_omega_earth},
  {"deltat", "const float", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->deltat},
  {"d_A_array_rotation", "float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_A_array_rotation},
  {"d_B_array_rotation", "float *", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->d_B_array_rotation},
  {"NSPEC_OUTER_CORE", "const int", (size_t) &((struct outer_core_impl_kernel_forward_args_s *) NULL)->NSPEC_OUTER_CORE}
};

struct param_info_s prepare_boundary_accel_on_device_params[] = {
  {"d_accel", "const float *", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->d_accel},
  {"d_send_accel_buffer", "float *", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->d_send_accel_buffer},
  {"num_interfaces", "const int", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->num_interfaces},
  {"max_nibool_interfaces", "const int", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->max_nibool_interfaces},
  {"d_nibool_interfaces", "const int *", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->d_nibool_interfaces},
  {"d_ibool_interfaces", "const int *", (size_t) &((struct prepare_boundary_accel_on_device_args_s *) NULL)->d_ibool_interfaces}
};

struct param_info_s prepare_boundary_potential_on_device_params[] = {
  {"d_potential_dot_dot_acoustic", "const float *", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->d_potential_dot_dot_acoustic},
  {"d_send_potential_dot_dot_buffer", "float *", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->d_send_potential_dot_dot_buffer},
  {"num_interfaces", "const int", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->num_interfaces},
  {"max_nibool_interfaces", "const int", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->max_nibool_interfaces},
  {"d_nibool_interfaces", "const int *", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->d_nibool_interfaces},
  {"d_ibool_interfaces", "const int *", (size_t) &((struct prepare_boundary_potential_on_device_args_s *) NULL)->d_ibool_interfaces}
};

struct param_info_s update_accel_acoustic_kernel_params[] = {
  {"accel", "float *", (size_t) &((struct update_accel_acoustic_kernel_args_s *) NULL)->accel},
  {"size", "const int", (size_t) &((struct update_accel_acoustic_kernel_args_s *) NULL)->size},
  {"rmass", "const float *", (size_t) &((struct update_accel_acoustic_kernel_args_s *) NULL)->rmass}
};

struct param_info_s update_accel_elastic_kernel_params[] = {
  {"accel", "float *", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->accel},
  {"veloc", "const float *", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->veloc},
  {"size", "const int", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->size},
  {"two_omega_earth", "const float", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->two_omega_earth},
  {"rmassx", "const float *", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->rmassx},
  {"rmassy", "const float *", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->rmassy},
  {"rmassz", "const float *", (size_t) &((struct update_accel_elastic_kernel_args_s *) NULL)->rmassz}
};

struct param_info_s update_disp_veloc_kernel_params[] = {
  {"displ", "float *", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->displ},
  {"veloc", "float *", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->veloc},
  {"accel", "float *", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->accel},
  {"size", "const int", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->size},
  {"deltat", "const float", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->deltat},
  {"deltatsqover2", "const float", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->deltatsqover2},
  {"deltatover2", "const float", (size_t) &((struct update_disp_veloc_kernel_args_s *) NULL)->deltatover2}
};

struct param_info_s update_potential_kernel_params[] = {
  {"potential_acoustic", "float *", (size_t) &((struct update_potential_kernel_args_s *) NULL)->potential_acoustic},
  {"potential_dot_acoustic", "float *", (size_t) &((struct update_potential_kernel_args_s *) NULL)->potential_dot_acoustic},
  {"potential_dot_dot_acoustic", "float *", (size_t) &((struct update_potential_kernel_args_s *) NULL)->potential_dot_dot_acoustic},
  {"size", "const int", (size_t) &((struct update_potential_kernel_args_s *) NULL)->size},
  {"deltat", "const float", (size_t) &((struct update_potential_kernel_args_s *) NULL)->deltat},
  {"deltatsqover2", "const float", (size_t) &((struct update_potential_kernel_args_s *) NULL)->deltatsqover2},
  {"deltatover2", "const float", (size_t) &((struct update_potential_kernel_args_s *) NULL)->deltatover2}
};

struct param_info_s update_veloc_acoustic_kernel_params[] = {
  {"veloc", "float *", (size_t) &((struct update_veloc_acoustic_kernel_args_s *) NULL)->veloc},
  {"accel", "const float *", (size_t) &((struct update_veloc_acoustic_kernel_args_s *) NULL)->accel},
  {"size", "const int", (size_t) &((struct update_veloc_acoustic_kernel_args_s *) NULL)->size},
  {"deltatover2", "const float", (size_t) &((struct update_veloc_acoustic_kernel_args_s *) NULL)->deltatover2}
};

struct param_info_s update_veloc_elastic_kernel_params[] = {
  {"veloc", "float *", (size_t) &((struct update_veloc_elastic_kernel_args_s *) NULL)->veloc},
  {"accel", "const float *", (size_t) &((struct update_veloc_elastic_kernel_args_s *) NULL)->accel},
  {"size", "const int", (size_t) &((struct update_veloc_elastic_kernel_args_s *) NULL)->size},
  {"deltatover2", "const float", (size_t) &((struct update_veloc_elastic_kernel_args_s *) NULL)->deltatover2}
};

struct param_info_s write_seismograms_transfer_from_device_kernel_params[] = {
  {"number_receiver_global", "const int *", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->number_receiver_global},
  {"ispec_selected_rec", "const int *", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->ispec_selected_rec},
  {"ibool", "const int *", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->ibool},
  {"station_seismo_field", "float *", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->station_seismo_field},
  {"d_field", "const float *", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->d_field},
  {"nrec_local", "const int", (size_t) &((struct write_seismograms_transfer_from_device_kernel_args_s *) NULL)->nrec_local}
};

struct param_info_s write_seismograms_transfer_strain_from_device_kernel_params[] = {
  {"number_receiver_global", "const int *", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->number_receiver_global},
  {"ispec_selected_rec", "const int *", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->ispec_selected_rec},
  {"ibool", "const int *", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->ibool},
  {"station_strain_field", "float *", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->station_strain_field},
  {"d_field", "const float *", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->d_field},
  {"nrec_local", "const int", (size_t) &((struct write_seismograms_transfer_strain_from_device_kernel_args_s *) NULL)->nrec_local}
};


struct  kernel_lookup_s {
  void *address;
  const char *name;
  size_t nb_params;
  struct param_info_s *params;
} function_lookup_table[] = { 
  {(void *) 0x4e3795, "assemble_boundary_accel_on_device", 6, assemble_boundary_accel_on_device_params},
  {(void *) 0x4e39b5, "assemble_boundary_potential_on_device", 6, assemble_boundary_potential_on_device_params},
  {(void *) 0x4ebb68, "compute_acoustic_kernel", 20, compute_acoustic_kernel_params},
  {(void *) 0x4e4351, "compute_add_sources_adjoint_kernel", 7, compute_add_sources_adjoint_kernel_params},
  {(void *) 0x4e45bc, "compute_add_sources_kernel", 8, compute_add_sources_kernel_params},
  {(void *) 0x4eb482, "compute_ani_kernel", 15, compute_ani_kernel_params},
  {(void *) 0x4eeb91, "compute_ani_undo_att_kernel", 21, compute_ani_undo_att_kernel_params},
  {(void *) 0x4e4ed6, "compute_coupling_CMB_fluid_kernel", 14, compute_coupling_CMB_fluid_kernel_params},
  {(void *) 0x4e5286, "compute_coupling_ICB_fluid_kernel", 14, compute_coupling_ICB_fluid_kernel_params},
  {(void *) 0x4e4884, "compute_coupling_fluid_CMB_kernel", 10, compute_coupling_fluid_CMB_kernel_params},
  {(void *) 0x4e4b5c, "compute_coupling_fluid_ICB_kernel", 10, compute_coupling_fluid_ICB_kernel_params},
  {(void *) 0x4e5544, "compute_coupling_ocean_kernel", 8, compute_coupling_ocean_kernel_params},
  {(void *) 0x4eb6f8, "compute_hess_kernel", 6, compute_hess_kernel_params},
  {(void *) 0x4eb075, "compute_iso_kernel", 16, compute_iso_kernel_params},
  {(void *) 0x4ef0db, "compute_iso_undo_att_kernel", 22, compute_iso_undo_att_kernel_params},
  {(void *) 0x4eacd0, "compute_rho_kernel", 6, compute_rho_kernel_params},
  {(void *) 0x4e67d2, "compute_stacey_acoustic_backward_kernel", 12, compute_stacey_acoustic_backward_kernel_params},
  {(void *) 0x4e6470, "compute_stacey_acoustic_kernel", 17, compute_stacey_acoustic_kernel_params},
  {(void *) 0x4e6f86, "compute_stacey_elastic_backward_kernel", 12, compute_stacey_elastic_backward_kernel_params},
  {(void *) 0x4e6c14, "compute_stacey_elastic_kernel", 19, compute_stacey_elastic_kernel_params},
  {(void *) 0x4ebead, "compute_strength_noise_kernel", 10, compute_strength_noise_kernel_params},
  {(void *) 0x4ee2d2, "crust_mantle_impl_kernel_adjoint", 80, crust_mantle_impl_kernel_adjoint_params},
  {(void *) 0x4eced6, "crust_mantle_impl_kernel_forward", 80, crust_mantle_impl_kernel_forward_params},
  {(void *) 0x4e3f9c, "get_maximum_scalar_kernel", 3, get_maximum_scalar_kernel_params},
  {(void *) 0x4e4128, "get_maximum_vector_kernel", 3, get_maximum_vector_kernel_params},
  {(void *) 0x4ea7c3, "inner_core_impl_kernel_adjoint", 62, inner_core_impl_kernel_adjoint_params},
  {(void *) 0x4e98f7, "inner_core_impl_kernel_forward", 62, inner_core_impl_kernel_forward_params},
  {(void *) 0x4e5da6, "noise_add_source_master_rec_kernel", 6, noise_add_source_master_rec_kernel_params},
  {(void *) 0x4e6085, "noise_add_surface_movie_kernel", 11, noise_add_surface_movie_kernel_params},
  {(void *) 0x4e5b90, "noise_transfer_surface_to_host_kernel", 5, noise_transfer_surface_to_host_kernel_params},
  {(void *) 0x4e8ba0, "outer_core_impl_kernel_adjoint", 36, outer_core_impl_kernel_adjoint_params},
  {(void *) 0x4e830c, "outer_core_impl_kernel_forward", 36, outer_core_impl_kernel_forward_params},
  {(void *) 0x4e3df5, "prepare_boundary_accel_on_device", 6, prepare_boundary_accel_on_device_params},
  {(void *) 0x4e3bd5, "prepare_boundary_potential_on_device", 6, prepare_boundary_potential_on_device_params},
  {(void *) 0x4e7a48, "update_accel_acoustic_kernel", 3, update_accel_acoustic_kernel_params},
  {(void *) 0x4e76d9, "update_accel_elastic_kernel", 7, update_accel_elastic_kernel_params},
  {(void *) 0x4e7203, "update_disp_veloc_kernel", 7, update_disp_veloc_kernel_params},
  {(void *) 0x4e746f, "update_potential_kernel", 7, update_potential_kernel_params},
  {(void *) 0x4e7bfb, "update_veloc_acoustic_kernel", 4, update_veloc_acoustic_kernel_params},
  {(void *) 0x4e78af, "update_veloc_elastic_kernel", 4, update_veloc_elastic_kernel_params},
  {(void *) 0x4e5773, "write_seismograms_transfer_from_device_kernel", 6, write_seismograms_transfer_from_device_kernel_params},
  {(void *) 0x4e5997, "write_seismograms_transfer_strain_from_device_kernel", 6, write_seismograms_transfer_strain_from_device_kernel_params}
};
