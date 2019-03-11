from __future__ import print_function
import os
import sys
import time
import numpy as np
import cPickle as cp

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
import gmr.gmr.gmm as gmm
import dataset as ds
import utils as utils

from sklearn.decomposition import PCA

do_pca = True

print('Loading image data...')
img_data = utils.extract_images(fname='bin/img_data_extend.pkl', only_digits=False)
print('Loading joint motion data...')
fa_data, fa_mean, fa_std = utils.extract_jnt_fa_parms(fname='bin/jnt_ik_fa_data_extend.pkl', only_digits=False)
fa_data_normed = (fa_data - fa_mean) / fa_std

#transform with PCA
if do_pca:
    pca_mdl = PCA(n_components=0.99)
    img_data_new = pca_mdl.fit_transform(img_data)
    print('Use PCA to reduce img data to {0} dim'.format(img_data_new.shape[1]))
    #save this model
    cp.dump(pca_mdl, open('img_pca_mdl.pkl', 'wb'))
    fa_data_new = pca_mdl.fit_transform(fa_data_normed)
    print('Use PCA to reduce fa_data_normed data to {0} dim'.format(fa_data_new.shape[1]))
    cp.dump(pca_mdl, open('fa_pca_mdl.pkl', 'wb'))
    raw_input()
    img_data = img_data_new
    fa_data_normed = fa_data_new

print('Constructing dataset...')
raw_input('ENTER to Start...')
#put them together
aug_data = np.concatenate((img_data, fa_data_normed), axis=1)


data_sets = ds.construct_datasets(aug_data, validation_ratio=.1, test_ratio=.1)

random_state = 0
# covariance_types = ['spherical', 'diag', 'tied', 'full']
covariance_types = ['full']
# covariance_types = ['diag']
n_comps = [10, 20, 50, 80, 100, 120]
# n_comps = [150, 180, 200, 250, 300, 350]
# n_comps = [380, 400, 430, 450, 480, 500]
bic_score_dict = {'spherical':[], 'diag':[], 'tied':[], 'full':[], 'n_comps':n_comps}
print('Start fitting...')
for covar_type in covariance_types:
    for n_comp in n_comps:
        gmm_mdl = gmm.GMM(n_components=n_comp, covariance_type=covar_type, random_state=random_state, verbose=2, n_iter=200, n_init=1)
        bic_score = gmm_mdl.fit(data_sets.train._data)

        print('BIC Score:'), bic_score

        mdl_name = 'gmm_{0}_comps{1}.pkl'.format(covar_type, n_comp)
        gmm_mdl.save_model(mdl_name)

        bic_score_dict[covar_type].append(bic_score)

bic_score_fname = 'gmm_bic_score_record.pkl'
cp.dump(bic_score_dict, open(bic_score_fname, 'wb'))
