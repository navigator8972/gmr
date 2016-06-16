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

print 'Loading image data...'
img_data = utils.extract_images(fname='bin/img_data_extend.pkl', only_digits=False)
print 'Loading joint motion data...'
fa_data, fa_mean, fa_std = utils.extract_jnt_fa_parms(fname='bin/jnt_ik_fa_data_extend.pkl', only_digits=False)
fa_data_normed = (fa_data - fa_mean) / fa_std

print 'Constructing dataset...'
#put them together
aug_data = np.concatenate((img_data, fa_data_normed), axis=1)

data_sets = ds.construct_datasets(aug_data, validation_ratio=.1, test_ratio=.1)

random_state = 0
covariance_types = ['spherical', 'diag', 'tied', 'full']
n_comps = [2]
bic_score_dict = {'spherical':[], 'diag':[], 'tied':[], 'full':[], 'n_comps':n_comps}
print 'Start fitting...'
for covar_type in covariance_types:
    for n_comp in n_comps:
        gmm_mdl = gmm.GMM(n_components=n_comp, covariance_type=covar_type, random_state=random_state, verbose=2, n_iter=10, n_init=1)
        bic_score = gmm_mdl.fit(data_sets.train._data)

        print 'BIC Score:', bic_score

        mdl_name = 'gmm_{0}_comps{1}.pkl'.format(covar_type, n_comp)
        gmm_mdl.save_model(mdl_name)

        bic_score_dict[covar_type].append(bic_score)

bic_score_fname = 'gmm_bic_score_record.pkl'
cp.dump(bic_score_dict, open(bic_score_fname, 'wb'))
