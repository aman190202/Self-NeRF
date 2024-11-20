from configargparse import ArguementParser
from load_llff import load_llff_data
import numpy as np
from nerf import NeRF,positional_encoding
import torch


def train():
    parser = ArguementParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    
    args = parser.parse_args()

    # Only meant for LLFF scenes
    if args.dataset_type == 'llff':

        # Let us not touch the dataloaders, these basically recall all the neccessary data from the files into our program
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor, # factor tells us if we have to sample down the images, which we should if our resolution is too high
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)

        H, W, focal = poses[0,:3,-1] # height width focal length
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])
        poses = poses[:,:3,:4] # exrtrinsic matrix paramters

        print('Loaded llff', images.shape, render_poses.shape)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:  # if args.llffhold = 5, we use every 5th image as our test image
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        # near and far bounding planes of the rendering space
        print('DEFINING BOUNDS') 
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.

        print('NEAR FAR', near, far)



    pass


if __name__=='__main__':
    train()