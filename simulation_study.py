import utils
import scipy.io as sio
import numpy as np

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
    def __init__(self, mat_or_matfile):
        if type(mat_or_matfile) is str:
            mat = sio.loadmat(mat_or_matfile)
        else:
            mat = mat_or_matfile
        self.mat = mat

    def electrons_in_chamber(self, bspacing=25e-9):
        """
        xx: bunch passages
        yy: number of electrons in chamber
        """
        xx = self.mat['t'][0,:]/bspacing
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

