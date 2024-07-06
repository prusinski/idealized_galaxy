# Based on: https://trident.readthedocs.io/en/latest/ion_balance.html
# Going to generate ions using Trident

import yt
import trident
fn = '../runs/m12i_res450000/output/snapshot_570.hdf5'
ds = yt.load(fn)
trident.add_ion_fields(ds, ions=['O VI'])
proj = yt.ProjectionPlot(ds, "z", "O_p5_number_density")
proj.save()
