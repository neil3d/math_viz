import plot_LTCs
import plot_spherical_gaussians

# configuration
plot_output_path = '../docs/images/plots'

# save figures
plot_LTCs.save_figure(plot_output_path)
plot_spherical_gaussians.save_figure(plot_output_path)

# save animations
plot_LTCs.save_animation(plot_output_path)
