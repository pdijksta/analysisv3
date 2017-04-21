from __future__ import division
import numpy as np
import cPickle as pickle

def id_keys(dd, identifiers, verbose=False):
    """
    dict, identifiers
    """
    if verbose: print(identifiers)
    id_keys = {}
    for id_ in identifiers:
        value = dd.keys()
        id_keys[id_] = sorted(value)
        if verbose: print(id_keys[id_])
        dd = dd[value[0]]
    return id_keys

def create_lists(dict_, var_arr):
    var_arr = map(str, var_arr)
    for ctr, var in enumerate(var_arr):
        if var == 'VAR':
            keys = sorted(dict_.keys())
            xx, yy = [], []
            for key in keys:
                fail = False
                this_dd = dict_[key]
                for var in var_arr[ctr+1:]:
                    if var == 'PASS' and len(this_dd) == 1:
                        this_dd = this_dd[this_dd.keys()[0]]
                    elif var =='PASS' and len(this_dd) != 1:
                        print(var_arr)
                        print(this_dd.keys())
                        raise ValueError('Illegal use of PASS')
                    else:
                        try:
                            this_dd = this_dd[var]
                        except:
                            print('Warning: Fail for %s, %s' % (key, var))
                            print(this_dd.keys())
                            fail = True
                            break
                if not fail:
                    yy.append(this_dd)
                    xx.append(key)

            if not xx:
                print var_arr
                print keys
                raise ValueError('Empty xx')

            try: xx = np.array(xx, dtype=float)
            except: pass
            try: yy = np.array(yy, dtype=float)
            except: pass

            return xx, yy

        elif var == 'PASS':
            keys = dict_.keys()
            if len(keys) != 1:
                print(keys)
                raise ValueError('Wrong use of PASS')
            dict_ = dict_[keys[0]]
        else:
            try:
                dict_ = dict_[var]
            except:
                import pdb ; pdb.set_trace()

    return [dict_], [42]

def add_up_beams(xx1, yy1, xx2, yy2):
    xx, yy = [], []
    ctr1, ctr2 = 0, 0
    x1, x2 = xx1[ctr1], xx2[ctr2]
    while True:
        if x1 == x2:
            xx.append(x1)
            yy.append(yy1[ctr1]+yy2[ctr2])
            ctr1 += 1
            ctr2 += 1
        elif x1 < x2:
            xx.append(x1)
            yy.append(2*yy1[ctr1])
            ctr1 += 1
        elif x1 > x2:
            xx.append(x2)
            yy.append(2*yy2[ctr2])
            ctr2 += 1

        if ctr1 == len(xx1):
            x1 = np.inf
        else:
            x1 = xx1[ctr1]
        if ctr2 == len(xx2):
            x2 = np.inf
        else:
            x2 = xx2[ctr2]

        if x1 == np.inf and x2 == np.inf:
            try: xx = np.array(xx, float)
            except: pass
            try: yy = np.array(yy, float)
            except: pass

            return xx,yy

def create_lists_beams(dict_, var_arr):
    var_arr1 = var_arr[:]
    var_arr1[var_arr.index('BEAMS')] = 'B1'
    var_arr2 = var_arr[:]
    var_arr2[var_arr.index('BEAMS')] = 'B2'

    xx1, yy1 = create_lists(dict_, var_arr1)
    xx2, yy2 = create_lists(dict_, var_arr2)

    xx, yy = add_up_beams(xx1, yy1, xx2, yy2)

    return xx, yy


device_title_dict = {
        'ArcDipReal' : 'Dipole',
        'ArcQuadReal' : 'Quadrupole',
        'Drift': 'Drift'
        }

def load_pkl(f):
    with open(f) as f:
        return pickle.load(f)
