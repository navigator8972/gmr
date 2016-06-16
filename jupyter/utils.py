"""
module of utilities
"""
import numpy as np
import cPickle as cp
from collections import defaultdict

import dataset as ds

def extract_images(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_image_helper(d):
        #flatten the image and scale them
        return d.flatten().astype(dtype) * 1./255.
    images = []
    if data_dict is not None:
        for char in sorted(data_dict.keys(), key=lambda k:k[-1]):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                images += [extract_image_helper(d) for d in data_dict[char]]
    return np.array(images)

def extract_jnt_trajs(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    def extract_jnt_trajs_helper(d):
        #flatten the image and scale them, is it necessary for joint trajectory, say within pi radians?
        return d.flatten().astype(dtype)
    jnt_trajs = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                jnt_trajs += [extract_jnt_trajs_helper(d) for d in data_dict[char]]
    return np.array(jnt_trajs)

def extract_jnt_fa_parms(data=None, fname=None, only_digits=True, dtype=np.float32):
    data_dict = data
    if fname is not None:
        #try to load from given fname
        data_dict = cp.load(open(fname, 'rb'))

    fa_parms = []
    if data_dict is not None:
        for char in sorted(data_dict.keys()):
            if only_digits and ord(char[-1]) > 57:
                continue
            else:
                fa_parms += [d for d in data_dict[char]]
    fa_parms = np.array(fa_parms)
    #Gaussian statistics for potential normalization
    fa_mean = np.mean(fa_parms, axis=0)
    fa_std = np.std(fa_parms, axis=0)
    return fa_parms, fa_mean, fa_std

