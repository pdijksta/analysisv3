from __future__ import division
import simulation_study as ss
import utils
import numpy as np
import matplotlib.pyplot as plt

import LHCMeasurementTools.mystyle as ms

ms.mystyle()

plt.close('all')

path_pkl = utils.load_pkl('/storage/cell_0/simulations/paths_matfiles_pyecloud.pkl')

#test_mat = '/storage/cell_0/simulations/LHC_MQ_6500GeV_sey1.40_1.1e11ppb_1/Pyecltest.mat'
test_mat = path_pkl['MB']['6500']['1.40']['1.1']['1']
study = ss.simulation_from_path(test_mat)

yy = study.mat['lam_t_array'][0,:]
xx = study.mat['t'][0,:] / 25e-9

fig = ms.figure('Heat loads of simulation')

sp = plt.subplot(2,2,1)
sp.grid(True)
sp.set_title('Filling pattern')
sp.set_xlabel('Bunch passages')
sp.set_ylabel('Beam charge density')
sp.plot(xx, yy)


first_train = 4*80+15

hl_details = study.heatload_rescaled(first_train, 2748, details=True)
hl_tot = hl_details['hl']
factor2 = hl_details['factor2']

sp = plt.subplot(2,2,2)
sp.grid(True)
sp.set_title('Heat load from e-cloud: %.2f\nMultiplicative factor for second bunch: %.2f' % (hl_tot, factor2))
sp.set_xlabel('Bunch passages')
sp.set_ylabel('Heat load from e-cloud [W/m]')

xx, yy = study.heatload_passage()
sp.plot(xx, yy)

sp.axvline(first_train, label='End of first train', color='black')
sp.axhline(np.mean(yy), label='Average HL', color='red')

sp.legend()


plt.show()

