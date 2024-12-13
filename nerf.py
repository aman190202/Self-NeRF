import torch
from torch import nn
import torch.nn.functional as F
import os

def positional_encoding(inputs, num_freqs=10):
    encoded = [inputs] 
    freq_bands = torch.arange(1, num_freqs + 1, dtype=torch.float32)
    for freq in freq_bands:
        encoded.append(torch.sin(inputs * freq))
        encoded.append(torch.cos(inputs * freq))
    return torch.cat(encoded, dim=-1)

def nn_chunks(nn, chunk):
    """Defines 'nn' for input chunks"""
    if chunk is None:
        return nn

    def ret(inputs):
        return torch.concat([nn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, nn, embed, embeddirs, netchunk=1024*64):
    """Applies neural network 'nn' to inputs"""

    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed(inputs_flat)
    if viewdirs is not None:
        input_dirs = torch.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs(input_dirs_flat)
        embedded = torch.concat([embedded, embedded_dirs], -1)

    outputs_flat = nn_chunks(nn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):

        super(NeRF, self).__init__()
        self.D = D # How dense the neural network is ( how many hidden layers )
        self.W = W # How wide the network is (256 in the official implementation)
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
        
        self.feature_linear = nn.Linear(W, W)
        self.alpha_linear = nn.Linear(W, 1)
        self.rgb_linear = nn.Linear(W//2, 3)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], -1)
    
        for i, _ in enumerate(self.views_linears):
            h = self.views_linears[i](h)
            h = F.relu(h)

        rgb = self.rgb_linear(h)
        outputs = torch.cat([rgb, alpha], -1)

        return outputs  


def create_nerf(args):
    """Instantiate NeRF's MLP model without hierarchical sampling."""
    # Hard-code multires to 10
    multires = 10

    # Define the positional encoding function
    def get_embed_fn(multires, include_input=True):
        def embed_fn(x):
            return positional_encoding(x, num_freqs=multires, include_input=include_input)
        
        input_dim = 3 * (2 * multires + (1 if include_input else 0))
        return embed_fn, input_dim

    # Get positional encoding for input and views
    embed_fn, input_ch = get_embed_fn(multires)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embed_fn(multires=4)  # Hard-code view positional encoding to 4

    # Define the NeRF model
    output_ch = 4  # Output: RGB (3) + alpha (1)
    skips = [4]  # Skip connections at layer 4
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(args.device)

    # Collect parameters for optimization
    grad_vars = list(model.parameters())

    # Define the network query function
    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    # Initialize experiment details
    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################
    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [
            os.path.join(basedir, expname, f)
            for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f
        ]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

    ##########################

    # Define rendering configurations
    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # Handle non-NDC settings for datasets
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    # Test-time rendering does not use perturbation or noise
    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer