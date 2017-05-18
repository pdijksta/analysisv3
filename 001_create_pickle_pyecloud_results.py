import os
import cPickle
import re
import argparse

import scipy.io as sio
import numpy as np
from scipy.constants import e as const_e
from scipy.io.matlab.miobase import MatReadError


# Argparse
parser = argparse.ArgumentParser(description='This script gathers heatloads from pyecload simulations and stores them in a pickled dict.')
parser.add_argument('-d', help='Delete old pickle and build completely new one. Default: Off.', action='store_true')
parser.add_argument('-f', help='Use regex for a fill simulation', action='store_true')
parser.add_argument('--emax', help='Use regex for a emax model simulation', action='store_true')
parser.add_argument('-s', help='Use regex for a s_param model simulation', action='store_true')
parser.add_argument('--musig', action='store_true')
parser.add_argument('--mu', action='store_true')
parser.add_argument('--theta', action='store_true')
parser.add_argument('--r0', action='store_true')
parser.add_argument('--ctr', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--sext', action='store_true')
parser.add_argument('--dipquad', action='store_true')
parser.add_argument('--substeps', action='store_true')
parser.add_argument('--cell', action='store_true')

parser.add_argument('dir', help='Directory with the simulations.', metavar='DIR')

args = parser.parse_args()
root_dir = args.dir

if not os.path.isdir(root_dir):
    raise ValueError('DIR is not a directory')

# Config
hl_pkl_name = root_dir + '/heatload_pyecloud3.pkl'
nel_hist_pkl_name = root_dir + '/nel_hist_pyecloud3.pkl'
path_pkl_name = root_dir + '/paths_matfiles_pyecloud.pkl'
fail_name = './fail_list.txt'

path_dict = {}

if args.d:
    hl_dict = {}
    nel_hist_dict = {}
else:
    if os.path.isfile(hl_pkl_name):
        with open(hl_pkl_name,'r') as f:
            hl_dict = cPickle.load(f)
    else:
        hl_dict = {}
    if os.path.isfile(nel_hist_pkl_name):
        with open(nel_hist_pkl_name,'r') as f:
            nel_hist_dict = cPickle.load(f)
    else:
        nel_hist_dict = {}

all_files = os.listdir(root_dir)
# Regular Expression for the folder names
if args.f:
    folder_re = re.compile('^Fill(\d+)_cut(\d+\.\d[1-9]*)0*h_\d+GeV_for_triplets_(B[1,2])_LHC_([A-Za-z]+)_\d+GeV_sey([\d\.]+)_coast([\d\.]+)$')
    identifiers = ['filln', 'time_of_interest', 'beam', 'device', 'sey', 'coast']
elif args.emax:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_Emax_(\d+)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'emax']
elif args.s:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_s_param(\d+\.\d+)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 's']
elif args.musig:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_ctr_(\d+)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'ctr']
elif args.mu:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_ctr_(\d\.\d)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'fact_mu']
elif args.theta:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_theta_(\d\.\d)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'theta']
elif args.r0:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_R0_(\d\.\d)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'r0']
elif args.ctr:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_ctr_(\d)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'ctr']
elif args.sext:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_([df])')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'df']
elif args.dipquad:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb')
    identifiers = ['device', 'energy', 'sey', 'intensity']
elif args.substeps:
    folder_re = re.compile('^LHC_([A-Za-z]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_f_substeps_(\d+)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'substeps']
elif args.cell:
    folder_re = re.compile('^LHC_([\w]+)_(\d+)GeV_sey(\d\.\d+)_(\d+\.\d+)e11ppb_(\d)')
    identifiers = ['device', 'energy', 'sey', 'intensity', 'photoemission']
else:
    raise ValueError('Regex not specified!')

fail_ctr = 0
success_ctr = 0

fail_lines = ''
fail_lines_IO = ''
const_LHC_frev = 11.2455e3

# Functions
def insert_to_nested_dict(dictionary, value, keys, must_enter=False, add_up=False):
    """
    Inserts value to nested dictionary. The location is specified by keys.
    If must_enter is set to True, an error is raised if the entry is already present.
    """
    for key in keys[:-1]:
        if key not in dictionary:
            dictionary[key] = {}
        dictionary = dictionary[key]

    last_key = keys[-1]
    if last_key not in dictionary:
        dictionary[last_key] = value
    elif add_up:
        dictionary[last_key] += value
    elif must_enter:
        raise ValueError('Key %s already exists!' % last_key)

def check_if_already_exist(dictionary, keys):
    """
    Returns True if a nested dict with the keys in the correct order does exist.
    """
    try:
        for key in keys:
            dictionary = dictionary[key]
    except KeyError:
        return False
    else:
        return True

def sort_new_dict_recursively(dictionary):
    for key in dictionary:
        if type(dictionary[key]) is dict:
            sort_new_dict_recursively(dictionary[key])
        elif type(dictionary[key]) is np.ndarray:
            dictionary[key] = np.sort(dictionary[key], axis=0)
        else:
            raise ValueError('Unknown type!')

def assure_both_beams(dictionary):
    for key in dictionary:
        if type(dictionary[key]) is dict:
            assure_both_beams(dictionary[key])
        elif type(dictionary[key]) is np.ndarray:
            mask_one_beam = dictionary[key][:,2] == 0
            dictionary[key][mask_one_beam,1] *= 2
        else:
            raise ValueError('Unknown type!')

# Main loop
for folder in all_files:
    file_info = re.search(folder_re,folder)
    if file_info is None:
        print('Folder %s did not match the regex!' % folder)
        continue
    keys = list(file_info.groups())

    id_dict = {}
    for identifier, info in zip(identifiers, keys):
        id_dict[identifier] = info
    print(keys)


    if not args.d and check_if_already_exist(hl_dict, keys):
        print('Continuing for', keys)
        continue

    mat_str = root_dir + '/' + folder + '/Pyecltest.mat'
    if not os.path.isfile(mat_str):
        print('Warning: file %s does not exist' % mat_str)
        fail_ctr += 1
        fail_lines += folder + '\n'
        continue

    print('Trying to read %s.' % mat_str)
    try:
        matfile = sio.loadmat(mat_str)
    except (IOError, MatReadError):
        print('IOError')
        fail_ctr += 1
        fail_lines_IO += folder + '\n'
        continue
    else:
        success_ctr += 1

    heatload = np.sum(matfile['energ_eV_impact_hist'])*const_LHC_frev*const_e
    e_transverse_hist = np.sum(matfile['nel_hist'],axis=0)


    insert_to_nested_dict(hl_dict, heatload, keys)
    insert_to_nested_dict(nel_hist_dict, e_transverse_hist, keys, must_enter=True)
    insert_to_nested_dict(path_dict, mat_str, keys, must_enter=True)

# add xg_hist variable only once
    if 'xg_hist' not in nel_hist_dict:
        insert_to_nested_dict(nel_hist_dict, matfile['xg_hist'][0], ['xg_hist'], must_enter=True)

#Sort new style dict

with open(hl_pkl_name, 'w') as pkl_file:
    cPickle.dump(hl_dict, pkl_file, -1)

with open(nel_hist_pkl_name, 'w') as pkl_file:
    cPickle.dump(nel_hist_dict, pkl_file, -1)

with open(path_pkl_name, 'w') as pkl_file:
    cPickle.dump(path_dict, pkl_file, -1)

print('%i simulations were successful and %i failed.' % (success_ctr,fail_ctr))
print(fail_lines)
print('IO')
print(fail_lines_IO)

#with open(fail_name,'w') as fail_file:
#    fail_file.write(fail_lines)
#    fail_file.write(fail_lines_IO)
#
