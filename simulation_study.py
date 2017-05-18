from __future__ import division
import imp
import os
import utils

import scipy.io as sio
from scipy.constants import e
import numpy as np

const_LHC_frev = 11.2455e3

class heatload_study(object):
    def __init__(self, pkl_file, identifiers):
        self.dictionary = utils.load_pkl(pkl_file)
        self.identifiers = identifiers
        self.id_keys = utils.id_keys(self.dictionary, self.identifiers)

    def create_lists(self, *keys):
        return utils.create_lists(self.dictionary, keys)

    def create_lists_beams(self, *keys):
        return utils.create_lists_beams(self.dictionary, keys)

class simulation(object):
    def __init__(self, mat_or_matfile, Dt=1e-11, dec_fact_out=13, b_spac=25e-9, filling_pattern=None):
        if type(mat_or_matfile) is str:
            mat = sio.loadmat(mat_or_matfile)
        else:
            mat = mat_or_matfile
        self.mat = mat
        self.Dt = Dt
        self.dec_fact_out = dec_fact_out
        self.b_spac = b_spac
        if filling_pattern:
            self.filling_pattern = np.array(filling_pattern, dtype=float)

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

    def heatload_passage(self):
        """
        xx: Bunch passages
        yy: Heat load scaled with the revolution frequency of the LHC and in SI units
        """
        xx = self.mat['t'][0,:] / self.b_spac
        yy = self.mat['En_imp_eV_time'][0,:] * const_LHC_frev * e

        # sum to get hl per bunch passage, one point per bunch
        shrink_factor = int(len(xx)/xx[-1])
        yy_max = int(len(yy)/shrink_factor)*shrink_factor
        yy2 = np.sum(yy[:yy_max].reshape(yy_max/shrink_factor, shrink_factor), axis=1)
        xx2 = xx[::shrink_factor]

        return xx2[:len(yy2)], yy2 * len(yy2)
    def heatload_rescaled(self, first_train, bunches_rescaled, verbose=False, details=False):
        """
        first train: bunch number after which the first train is considered to have ended.
        the rest of the beam is supposed to be the second train.
        The heat load of the first train and the second train is rescaled to the total bunches,
        where the first tran is only considered once.
        """
        hl_arr = self.mat['En_imp_eV_time'][0,:] * const_LHC_frev * e
        t_arr = self.mat['t'][0,:]
        b_spac_arr = t_arr/ self.b_spac

        mask_first_bunch = b_spac_arr < first_train
        mask_second_bunch = b_spac_arr > first_train
        n_bunches_first = np.sum(self.filling_pattern[:first_train])
        n_bunches_second = np.sum(self.filling_pattern[first_train:])

        factor2 = (bunches_rescaled - n_bunches_first)/n_bunches_second
        hl = np.sum(hl_arr[mask_first_bunch]) + np.sum(hl_arr[mask_second_bunch])*factor2

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


def simulation_from_path(path):
    directory = os.path.expanduser(os.path.abspath(os.path.dirname(path)))

    pyecltest = directory+'/Pyecltest.mat'
    beam_beam = imp.load_source('beam.beam', directory +'/beam.beam')
    #machine_parameters = imp.load_source('machine_parameters', directory +'/machine_parameters.input')
    simulation_parameters = imp.load_source('simulation_parameters', directory +'/simulation_parameters.input')
    sim = simulation(pyecltest,
                     Dt=simulation_parameters.Dt,
                     dec_fact_out=simulation_parameters.dec_fact_out,
                     b_spac=beam_beam.b_spac,
                     filling_pattern=beam_beam.filling_pattern_file)
    return sim

