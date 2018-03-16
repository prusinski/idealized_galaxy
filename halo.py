"""
Create a 3D (2D) grid of gas cells to represent the volume around a generic galaxy.
The grid exists in two parts:
1) A galaxy halo (sphere) at the center of the volume with an NFW density 
   profile and velocity vectors traveling radially outward from the center with
   uniform velocity.
2) A background IGM with uniform density beyond the virial radius of the halo
   and uniform velocity all traveling radially inward toward the center of the 
   halo.
"""
import yt 
import numpy as np

# Parameters to modify
resolution = 4096
box_width = 10 #Mpc
halo_radius = 70 #kpc
igm_vel = -80 #km/s negative means inflowing
halo_vel = 200 #km/s
igm_weight = 1/500 # relative weight to uniform IGM density versus halo density at r_vir

# initialize a 3D grid of uniform grid density (2D + 1 dimension in X)
arr = np.ones([1, resolution, resolution])
zeros = 0*arr
data = dict(test = (arr, 'g'))
bbox = np.array([[-box_width/2, box_width/2], [-box_width/2, box_width/2], 
                 [-box_width/2, box_width/2]]) # bounding box
# Create the dataset in yt
ds = yt.load_uniform_grid(data, arr.shape, length_unit="Mpc", bbox=bbox, nprocs=64)
ad = ds.all_data()
c = [0,0,0] # center

# Virial radius and scale radius.  Acceptable values for scale radius are 
# between 0.025 and 0.25 for actual galaxies fit with NFWs. We choose 0.25 to
# maximize mass in halo.
r_vir = ds.quan(halo_radius, 'kpc')
r_scale = (0.25 * r_vir).in_units('cm').v
r_scale3 = r_scale**3

# Add in additional fields: density, velocity, etc.
# Each field will have two components, one for the background IGM, the other
# for the halo.
def _density(field, data):
    halo = data['radius'] <= r_vir
    # uniform density beyond r_vir
    vals = igm_weight * data.ds.arr(data['ones'], 'g/cm**3') / (r_scale*r_vir.in_units('cm').v + r_scale3*r_vir.in_units('cm').v**3) 
    # NFW density for halo: r_scale is 25% of r_vir
    vals[halo] = 1/(r_scale*data['radius'][halo].v + (r_scale3*data['radius'][halo]**3).v)
    return vals
ds.add_field(("gas", "density"), function=_density, units="g/cm**3", sampling_type='cell')

# x velocity component
def _x_velocity(field, data):
    halo = data['radius'] <= r_vir
    vals = data.ds.arr(igm_vel*(data['x'] / data["radius"]).v, 'km/s')
    vals[halo] = data.ds.arr(halo_vel*(data['x'][halo] / data["radius"][halo]).v, 'km/s')
    return vals
ds.add_field(("gas", "velocity_x"), function=_x_velocity, units="km/s", sampling_type='cell')

# y velocity component
def _y_velocity(field, data):
    halo = data['radius'] <= r_vir
    vals = data.ds.arr(igm_vel*(data['y'] / data["radius"]).v, 'km/s')
    vals[halo] = data.ds.arr(halo_vel*(data['y'][halo] / data["radius"][halo]).v, 'km/s')
    return vals
ds.add_field(("gas", "velocity_y"), function=_y_velocity, units="km/s", sampling_type='cell')

# z velocity component
def _z_velocity(field, data):
    halo = data['radius'] <= r_vir
    vals = data.ds.arr(igm_vel*(data['z'] / data["radius"]).v, 'km/s')
    vals[halo] = data.ds.arr(halo_vel*(data['z'][halo] / data["radius"][halo]).v, 'km/s')
    return vals
ds.add_field(("gas", "velocity_z"), function=_z_velocity, units="km/s", sampling_type='cell')

# total velocity magnitude
def _total_velocity(field, data):
    return np.sqrt(data['velocity_x']**2 + data['velocity_y']**2 + data['velocity_z']**2)
ds.add_field(("gas", "velocity_magnitude"), function=_total_velocity, units="km/s", sampling_type='cell')

# column density for a cell
def _column_density(field, data):
    return data['density'] * data['dz']
ds.add_field(("gas", "column_density"), function=_column_density, units="g/cm**2", sampling_type='cell')

# add an impact parameter field assuming center = 0,0,0
def _impact_parameter(field, data):
    return np.sqrt(data["x"]**2 + data["y"]**2)
ds.add_field(("index", "impact_parameter"), function=_impact_parameter, units="kpc", sampling_type='cell')

# Add a cosmological line of sight velocity field
H_2 = ds.quan(200, 'km/s/Mpc') 
def _v_cosmo(field, data):
    return H_2*data['z']
ds.add_field(("gas", "velocity_cosmo"), function=_v_cosmo, units="km/s", sampling_type='cell')

# Make an effective LOS velocity field, which is a combination of the true LOS
# velocity (z velocity) and the hubble velocity (H * D_z) = hubble parameter at 
# redshift 2 ~ 200 km/s * the distance in z direction from galaxy.
# in this case, it is somewhat simplified because the galaxy sits at the origin
# and has zero velocity.

# Remember there is a cross term because this is at z=2, so line of sight 
# velocity is actually 3*velocity_z + cosmological velocity
def _v_los(field, data):
    return 3*data['velocity_z'] + H_2*data['z']
ds.add_field(("gas", "velocity_los"), function=_v_los, units="km/s", sampling_type='cell')


# Plot a top-down view of the field in density and overlay the velocity vectors
s = yt.SlicePlot(ds, "x", ('gas', 'density'), center=c, data_source=ad, width=(10, 'Mpc'))
s.set_cmap("density", "Blues")
s.annotate_marker(c)
s.annotate_quiver('velocity_y', 'velocity_z', factor=12, plot_args={"color":"red"})
s.save('doppler.png')
s.annotate_clear()
s.annotate_marker(c)
s.annotate_quiver('zeros', 'velocity_cosmo', factor=12, plot_args={"color":"green"})
s.save('cosmo.png')
s.annotate_clear()
s.annotate_marker(c)
s.annotate_quiver('velocity_y', 'velocity_los', factor=12, plot_args={"color":"black"})
s.save('cosmo_doppler.png')

# make a phase plot comparing impact parameter to effective LOS velocity
phase = yt.PhasePlot(ad, ('index', "impact_parameter"), ('gas', "velocity_los"), ["column_density"], weight_field=None)
phase.set_log('velocity_los', False)
phase.set_xlim(20, 5000)
phase.set_ylim(-1500, 1500)
phase.set_cmap('column_density', 'dusk')
#phase.set_zlim('column_density', 1e-46, 1e-42)
phase.save('phase.png')
