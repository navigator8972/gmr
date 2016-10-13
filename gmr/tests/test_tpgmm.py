import numpy as np
import gmr.gmr.gmm as gmm
import gmr.gmr.tpgmm as tpgmm
from gmr.gmr.mvn import MVN

from scipy.io import loadmat

import matplotlib.pyplot as plt

def main():
    #load data: from Calinon's PbDlib
    data = loadmat('Data02.mat')
    #process the data to extract useful information
    n_samples = data['nbSamples'][0, 0]
    n_dims = len(data['Data'])
    n_frames = len(data['Data'][0])
    n_steps = len(data['Data'][0][0]) / n_samples

    frames = data['s']['p'][0]

    frames_processed = [[{'A':frames[s][0][f][0], 'b':frames[s][0][f][1].T[0]} for f in range(n_frames)] for s in range(n_samples)]
    print frames_processed
    data = np.array(data['Data'])
    print data.shape
    data_processed = []
    for i in range(n_samples):
        traj = np.copy(data[:, :, (i*n_steps):((i+1)*n_steps)])
        traj=np.rollaxis(traj, 2)
        traj=np.swapaxes(traj, 1, 2)
        traj = np.array([pnt.flatten() for pnt in traj])
        data_processed.append(traj)

    n_components = 3
    n_feature_lens = [3, 3]
    tpgmm_mdl = tpgmm.TPGMM(n_components, n_feature_lens, random_state=0)
    tpgmm_mdl.fit(np.concatenate(data_processed, axis=0))

    # tpgmm_mdl.priors = [0.3917, 0.2647, 0.3435]
    # tpgmm_mdl.means = [ [[0.5635, -0.0359, 0.8601], [0.8114, 0.7690, 1.7407], [1.6576, 2.3853, 3.1541]],
    #                     [[0.5635, 1.1509, 3.2869], [0.8114, -1.7191, 2.3750], [1.6576, -0.085, 0.5583]]
    #                     ]
    # tpgmm_mdl.covariances = [   [   [[0.1320,   -0.0153,   0.2374], [-0.0153,    0.0049,   -0.0181], [0.2374,   -0.0181,    0.6284]],
    #                                 [[0.1310,    0.2388,    0.3203], [0.2388,    1.0361,    0.1721], [0.3203,    0.1721,    1.1246]],
    #                                 [[0.0418,    0.0234,    0.0691], [0.0234,    1.5213,   -0.6281], [0.0691,   -0.6281,    0.7709]]
    #                             ],
    #                             [   [[0.1320,   -0.0837,   -0.2403], [-0.0837,    1.6709,   -0.6928], [-0.2403,   -0.6928,    0.9989]],
    #                                 [[0.1310,    0.3956,   -0.2300], [0.3956,    1.7637,   -0.4835], [-0.2300,   -0.4835,    0.5908]],
    #                                 [[0.0418,    0.0009,   -0.0838], [0.0009,    0.0019,    0.0030], [-0.0838,    0.0030,    0.1994]]
    #                             ]
    #                         ]
    #
    # tpgmm_mdl.gmms[0].priors = tpgmm_mdl.priors
    # tpgmm_mdl.gmms[0].means = tpgmm_mdl.means[0]
    # tpgmm_mdl.gmms[0].covariances = tpgmm_mdl.covariances[0]
    #
    # tpgmm_mdl.gmms[1].priors = tpgmm_mdl.priors
    # tpgmm_mdl.gmms[1].means = tpgmm_mdl.means[1]
    # tpgmm_mdl.gmms[1].covariances = tpgmm_mdl.covariances[1]
    # print tpgmm_mdl.priors
    # print tpgmm_mdl.means
    # print tpgmm_mdl.covariances
    #
    likelihoods = tpgmm_mdl.to_probability_density(np.concatenate(data_processed, axis=0))
    # print data_processed[0][0, :]
    # print likelihoods[0, :]
    print np.mean(np.log(np.sum(likelihoods+1e-14, axis=1)))

    # print pdf_multivariate_gauss(np.array([data_processed[0][0, 0:3]]).T, np.array([tpgmm_mdl.means[0][0]]).T, tpgmm_mdl.covariances[0][0])
    #plot the trajectories
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.hold(True)
    for traj, frames in zip(data_processed, frames_processed):
        traj_original = np.array([frames[0]['A'].dot(pnt[0:3])+frames[0]['b'] for pnt in traj])
        ax1.plot(traj_original[:, 1], traj_original[:, 2], 'k', linewidth=3.0)
        # ax1.plot([traj_original[0, 1]], [traj_original[0, 2]], 'bo')
        # ax1.plot([traj_original[-1, 1]], [traj_original[-1, 2]], 'go')
        ax1.plot([frames[0]['b'][1]], [frames[0]['b'][2]], 'bo')
        ax1.plot([frames[1]['b'][1]], [frames[1]['b'][2]], 'go')

    # ax2 = fig.add_subplot(122)
    # ax2.hold(True)
    # for traj in data_processed:
    #     ax2.plot(traj[:, 4], traj[:, 5], 'k', linewidth=3.0)
    #     ax2.plot([traj[0, 4]], [traj[0, 5]], 'bo')
    #     ax2.plot([traj[-1, 4]], [traj[-1, 5]], 'go')

    #reproduction
    rep_trajs = []
    t = np.arange(n_steps) * 0.01 + 0.01
    ax3 = fig.add_subplot(121)
    ax3.hold(True)
    for frames in frames_processed:
        mean, covar = tpgmm_mdl.predict(indices=[0], X=np.array([t]).T, transformations=frames)
        ax3.plot(mean[:, 0], mean[:, 1], 'r-')
        ax3.plot([frames[0]['b'][1]], [frames[0]['b'][2]], 'bo')
        ax3.plot([frames[1]['b'][1]], [frames[1]['b'][2]], 'go')

    #generalization
    #new frames by randomly mixing the existing frames
    new_frames = []
    for sample_idx in range(n_samples):
        new_frame = []
        for frame_idx in range(n_frames):
            #randomly select two frames
            mix_frame_indices = np.random.choice(range(n_samples), 2)
            weights = np.random.rand(2)
            weights = weights / np.sum(weights)
            new_frame.append({  'A':weights[0]*frames_processed[mix_frame_indices[0]][frame_idx]['A']+weights[1]*frames_processed[mix_frame_indices[1]][frame_idx]['A'],
                                'b':weights[0]*frames_processed[mix_frame_indices[0]][frame_idx]['b']+weights[1]*frames_processed[mix_frame_indices[1]][frame_idx]['b']})
        new_frames.append(new_frame)
    print 'new frames:'
    print new_frames
    new_rep_trajs = []
    ax2 = fig.add_subplot(122)
    ax2.hold(True)
    for frames in new_frames:
        mean, covar = tpgmm_mdl.predict(indices=[0], X=np.array([t]).T, transformations=frames)
        ax2.plot(mean[:, 0], mean[:, 1], 'r-')
        ax2.plot([frames[0]['b'][1]], [frames[0]['b'][2]], 'bo')
        ax2.plot([frames[1]['b'][1]], [frames[1]['b'][2]], 'go')
    plt.show()
    return

def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    # assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    # assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    # assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    # assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    # assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1. / ( ((2* np.pi)**(float(len(mu))/2)) * (np.linalg.det(cov)**(1./2)) )
    part2 = (-1./2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    print part1, part2
    return float(part1 * np.exp(part2))

if __name__ == '__main__':
    main()
