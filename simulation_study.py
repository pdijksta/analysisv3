import utils

class simulation_study(object):
    def __init__(self, pkl_file, identifiers):
        self.pkl = utils.load_pkl(pkl_file)
        self.identifiers = identifiers
        self.id_keys = utils.id_keys(self.pkl, self.identifiers)

    def create_lists(self, keys):
        return utils.create_lists(self.pkl, keys)

    def create_lists_beams(self, keys):
        return utils.create_lists_beams(self.pkl, keys)

