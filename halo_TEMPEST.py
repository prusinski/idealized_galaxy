import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
import yt
import sys

def _column_density(field, data):
    return data[('gas', 'H_number_density')] * data[('index', 'dz')]

yt.add_field(("gas","HI_column_density"), function=_column_density, units="cm**-2")

if __name__ == '__main__':

    fn_list = open(sys.argv[1], 'r')
    fns = fn_list.readlines()

    for fn in yt.parallel_objects(fns):
        fn = fn.strip()
        fn_head = fn.split('/')[-1]
        ds = yt.load(fn)
        _, c = ds.find_max('density')
        rvir = ds.quan(30, 'kpc')
        sp = ds.sphere(c, rvir)
        bulk_vel = sp.quantities.bulk_velocity()
        print("Bulk Velocity of Halo = %s" % bulk_vel.to('km/s'))
        sp.set_field_parameter("bulk_velocity", bulk_vel)
        ad = ds.all_data()
        ad.set_field_parameter('center', c)
        ad.set_field_parameter('normal', [0,0,1])
        ad.set_field_parameter('bulk_velocity', bulk_vel)
        p = yt.PhasePlot(ad, ('index', 'cylindrical_radius'), ('gas', 'velocity_cylindrical_z'), ('gas', 'HI_column_density'), weight_field=None)
        p.set_unit('cylindrical_radius', 'kpc')
        p.set_unit('velocity_cylindrical_z', 'km/s')
        p.set_log('velocity_cylindrical_z', False)
        p.set_xlim(1e1, 1e4)
        p.set_ylim(-1000,1000)
        p.save()
