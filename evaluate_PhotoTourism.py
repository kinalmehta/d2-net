
import os
import argparse
import imageio
from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

import torch
import numpy as np
import cv2

from lib.datasetGrid import PhotoTourism as PT
from lib.datasetGridGray import PhotoTourism as PTGray

import matplotlib.pyplot as plt


# Argument parsing
parser = argparse.ArgumentParser(description='Feature extraction script')

parser.add_argument(
    '--multiscale', dest='multiscale', action='store_true',
    help='extract multiscale features'
)
parser.set_defaults(multiscale=False)
args = parser.parse_args()


# CUDA
use_cuda = False # torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)

def evaluate_RGB(model_path, dataset):

    # Creating CNN model
    model = D2Net(
        model_file=model_path,
        use_relu=True,
        use_cuda=False
    )
    model = model.to(device)    

    n_matches = []
    n_feats = []
    t_err = {thr: 0 for thr in rng}

    for examp in dataset:
        igp1, igp2, pts1, pts2, H = examp['image1'], examp['image2'], examp['pos1'], examp['pos2'], np.array(examp['H'])
        
        # predicting keypoints and descriptors using the d2-net model
        # print(igp1.shape, igp2.shape)
        with torch.no_grad():
            if args.multiscale:
                keypoints1, scores1, descriptors1 = process_multiscale(
                    igp1.to(device).unsqueeze(0),
                    model
                )
                keypoints2, scores2, descriptors2 = process_multiscale(
                    igp2.to(device).unsqueeze(0),
                    model
                )
            else:
                keypoints1, scores1, descriptors1 = process_multiscale(
                    igp1.to(device).unsqueeze(0),
                    model,
                    scales=[1]
                )
                keypoints2, scores2, descriptors2 = process_multiscale(
                    igp2.to(device).unsqueeze(0),
                    model,
                    scales=[1]
                )
        n_feats.append(keypoints1.shape[0])
        n_feats.append(keypoints2.shape[0])
        # applying nearest neighbor to find matches
        matches = mnn_matcher(
                torch.from_numpy(descriptors1).to(device=device), 
                torch.from_numpy(descriptors2).to(device=device)
            )

        pos_a = keypoints1[matches[:, 0], : 2] 
        pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
        pos_b_proj_h = np.transpose(np.dot(H, np.transpose(pos_a_h)))
        pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2 :]
        
        pos_b = keypoints2[matches[:, 1], : 2]
        
        dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))
        
        n_matches.append(matches.shape[0])
        
        if dist.shape[0] == 0:
            dist = np.array([float("inf")])

        for thr in rng:
            t_err[thr] += np.mean(dist <= thr)

    return t_err, np.array(n_feats), np.array(n_matches)


def evaluate(dataset_path):

    names = ['D2-Net Caffe Base', 'D2-Net Caffe Trained']
    pp = ['caffe', 'caffe']
    colors = ['red', 'green', 'blue', 'yellow']
    linestyles = ['-', '-', '-', '-']

    models_to_test = ['models/d2_ots.pth', 'models/d2_tf.pth']
    dataset = PT(dataset_path, preprocessing='caffe')
    dataset.build_dataset(cropSize=256)
    results = []
    for model_path, pre_process in zip(models_to_test,pp):
        dataset.preprocessing = pre_process
        results.append(evaluate_RGB(model_path, dataset))

    # plt the results
    plt_lim = [1, 10]
    plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
    plt.rc('axes', titlesize=25)
    plt.rc('axes', labelsize=25)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 1, 1)
    for result, name, color, linestyle in zip(results, names, colors, linestyles):
        t_err, feats, matches = result
        plt.plot(plt_rng, [t_err[thr] for thr in plt_rng], color=color, ls=linestyle, label=name, linewidth=3)

    plt.title('Overall')
    plt.xlim(plt_lim)
    plt.xticks(plt_rng)
    plt.ylabel('MMA')
    plt.ylim([0, 1])
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.legend()
    plt.show()
    plt.savefig('test_PT.png', bbox_inches='tight', dpi=300)

if __name__=="__main__":
    dataset_path = "../../dataset/phototourism/brandenburg_gate/dense/images/"
    evaluate(dataset_path)
