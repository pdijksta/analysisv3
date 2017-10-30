from __future__ import division
import os

import scipy.io as sio
from scipy.constants import e, c
import numpy as np

import utils

import HeatLoadCalculators.impedance_heatload as hli
import HeatLoadCalculators.synchrotron_radiation_heatload as hls

imp_calc = hli.HeatLoadCalculatorImpedanceLHCArc()
sr_calc = hls.HeatLoadCalculatorSynchrotronRadiationLHCArc()

const_LHC_frev = 11.2455e3
const_len_cryogenic_cell = 53.45

class heatload_study(object):
    def __init__(self, pkl_file, identifiers, title=None):
        """
        pkl_file, identifiers, title
        """
        if type(pkl_file) is str:
            self.dictionary = utils.load_pkl(pkl_file)
        else:
            self.dictionary = pkl_file
        self.identifiers = identifiers
        self.id_keys = utils.id_keys(self.dictionary, self.identifiers)
        self.title = title

    def create_lists(self, *keys, **kwargs):
        return utils.create_lists(self.dictionary, keys, **kwargs)

    def create_lists_beams(self, *keys):
        return utils.create_lists_beams(self.dictionary, keys)

    def get_first_entry(self):
        keys = ['PASS'] * len(self.identifiers)
        return utils.create_lists(self.dictionary, keys, expert=True)[0][0]

    def create_lists_path(self, func_name, func_args, func_kwargs, *keys, **kwargs):
        xx, paths = self.create_lists(*keys, **kwargs)
        yy = []
        for path in paths:
            sim = simulation_from_path(path)
            function = getattr(sim, func_name)
            yy.append(function(*func_args, **func_kwargs))
        return xx, np.array(yy)

simulation_study = heatload_study


class simulation_general(object):
    def electrons_in_chamber(self):
        """
        xx: bunch passages
        yy: number of electrons in chamber
        """
        xx = self.mat['t'][0,:]/self.b_spac
        yy = self.mat['Nel_timep'][0,:]
        return xx, yy

    def electrons_total_from_hist(self):
        """
        xx: bunch passages
        yy: number of electrons in chamber
        """
        yy = np.sum(self.mat['nel_hist'], axis=1)
        xx = np.array(xrange(0, len(yy)), dtype=float)
        return xx, yy

    def heatload_passage(self, b_spac=None):
        """
        xx: Bunch passages
        yy: Heat load scaled with the revolution frequency of the LHC and in SI units
        """
        if b_spac is None:
            b_spac = self.b_spac

        xx = self.mat['t'][0,:] / b_spac
        yy = self.mat['En_imp_eV_time'][0,:] * const_LHC_frev * e

        # sum to get hl per bunch passage, one point per bunch
        shrink_factor = int(len(xx)/xx[-1])
        yy_max = int(len(yy)/shrink_factor)*shrink_factor
        yy2 = np.sum(yy[:yy_max].reshape(int(yy_max/shrink_factor), shrink_factor), axis=1)
        xx2 = xx[::shrink_factor]

        return xx2[:len(yy2)], yy2

    def heatload_total(self):
        return np.sum(self.mat['En_imp_eV_time']) * const_LHC_frev * e

    def angle_hist_total(self):
        """
        Use with plt.step(xx, yy, where='mid')
        """
        yy = np.sum(self.mat['cos_angle_hist'], axis=0)
        xx = np.linspace(0, 1, len(yy))
        return xx, yy

    def kinetic_energy(self):
        xx = self.mat['t'][0,:] / self.b_spac
        yy = self.mat['En_kin_eV_time'][0,:]
        return xx, yy

    def central_density(self):
        xx = self.mat['t'][0,:] / self.b_spac
        yy = self.mat['cen_density'][0,:]
        return xx, yy

    def energy_impact_hist(self):
        xx = self.mat['xg_hist'][0,:]
        yy = np.sum(self.mat['energ_eV_impact_hist'], axis=0)
        return xx, yy



class simulation(simulation_general):
    def __init__(self, mat_or_matfile):
        if type(mat_or_matfile) is str:
            mat = sio.loadmat(mat_or_matfile)
        else:
            mat = mat_or_matfile
        self.mat = mat



class simulation_from_path(simulation_general):
    def __init__(self, path):
        directory = os.path.abspath(os.path.dirname(os.path.expanduser(path)))

        pyecltest = directory+'/Pyecltest.mat'
        beam_beam = utils.load_file_as_module(directory+'/beam.beam')
        machine_parameters = utils.load_file_as_module(directory+'/machine_parameters.input')
        simulation_parameters = utils.load_file_as_module(directory+'/simulation_parameters.input')
        secondary_emission_parameters = utils.load_file_as_module(directory+'/secondary_emission_parameters.input')

        self.mat = sio.loadmat(pyecltest)
        self. Dt = simulation_parameters.Dt,
        self.dec_fact_out = simulation_parameters.dec_fact_out,
        self.b_spac = beam_beam.b_spac,
        self.filling_pattern = np.array(beam_beam.filling_pattern_file)
        self.beam_beam = beam_beam
        self.machine_parameters = machine_parameters
        self.simulation_parameters = simulation_parameters
        self.secondary_emission_parameters = secondary_emission_parameters

    def heatload_rescaled(self, first_train, bunches_rescaled, verbose=False, details=False, double_hl=False):
        """
        first train: bunch number after which the first train is considered to have ended.
        the rest of the beam is supposed to be the second train.
        The heat load of the first train and the second train is rescaled to the total bunches,
        where the first train is only considered once.
        """
        hl_arr = self.mat['En_imp_eV_time'][0,:] * const_LHC_frev * e
        t_arr = self.mat['t'][0,:]
        b_spac_arr = t_arr/self.b_spac

        mask_first_bunch = b_spac_arr < first_train
        mask_second_bunch = b_spac_arr > first_train
        n_bunches_first = np.sum(self.filling_pattern[:first_train])
        n_bunches_second = np.sum(self.filling_pattern[first_train:])

        factor2 = (bunches_rescaled - n_bunches_first)/n_bunches_second
        hl1 = np.sum(hl_arr[mask_first_bunch])
        hl2 = np.sum(hl_arr[mask_second_bunch])

        hl = hl1 + hl2 * factor2
        if double_hl:
            hl *= 2

        if verbose or details:
            factor_alt = bunches_rescaled/(n_bunches_first+n_bunches_second)
            hl_alt = np.sum(hl_arr) * factor_alt
            if verbose:
                for key, value in locals().iteritems():
                    if '__' not in key:
                        print(key, value)

        if details:
            return locals()
        else:
            return hl

    def en_hist(self):
        yy = np.sum(self.mat['En_hist'], axis=0)
        xx = np.squeeze(self.mat['En_g_hist'])

        return xx, yy

    def calc_impedance_sr(self, bunches_rescaled=None, double_hl=False):
        """
        Arguments:
            bunches_rescaled: Number of filled bunches for which impedance should be calculated
            double_hl: Apply a factor 2 to the output.
        """
        bunch_int = np.array([self.beam_beam.fact_beam * self.filling_pattern])
        sigma_t = self.beam_beam.sigmaz/c
        fill_energy = self.beam_beam.energy_eV
        #n_bunches = np.sum(self.filling_pattern)
        imp = imp_calc.calculate_P_Wm(bunch_int, sigma_t, fill_energy) * const_len_cryogenic_cell
        sr = sr_calc.calculate_P_Wm(bunch_int, sigma_t, fill_energy) * const_len_cryogenic_cell

        if bunches_rescaled is None:
            factor_bunches = 1
        else:
            factor_bunches = bunches_rescaled/np.sum(self.filling_pattern)

        if double_hl:
            factor_double = 2
        else:
            factor_double = 1
        return imp*factor_bunches*factor_double, sr*factor_bunches*factor_double

