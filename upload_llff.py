import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data

def upload():

    
    # Load data
    K = None

    images, poses, bds, render_poses, i_test = load_llff_data("fern", 8,
                                                                recenter=True, bd_factor=.75,
                                                                spherify="store_true")
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, "datadir")
    if not isinstance(i_test, list):
        i_test = [i_test]

    # if .llffhold > 0:
    #     print('Auto LLFF holdout,', args.llffhold)
    i_test = np.arange(images.shape[0])[::8]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if True:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

    # elif args.dataset_type == 'blender':
    #     images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    #     print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     near = 2.
    #     far = 6.

    #     if args.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif args.dataset_type == 'LINEMOD':
    #     images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
    #     print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
    #     print(f'[CHECK HERE] near: {near}, far: {far}.')
    #     i_train, i_val, i_test = i_split

    #     if args.white_bkgd:
    #         images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    #     else:
    #         images = images[...,:3]

    # elif args.dataset_type == 'deepvoxels':

    #     images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
    #                                                              basedir=args.datadir,
    #                                                              testskip=args.testskip)

    #     print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split

    #     hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
    #     near = hemi_R-1.
    #     far = hemi_R+1.
    
        # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    # if args.render_test:
    #     render_poses = np.array(poses[i_test])