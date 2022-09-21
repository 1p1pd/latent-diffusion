import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0
                    ):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,)
    return samples, intermediates


def make_batch(image, img_path, device):
    if image is None:
        image = np.array(Image.open(img_path).convert("RGB"))
        image = image.astype(np.float32)/255.0
        image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)
        image = image[:, :, 272:528, 272:528]
    else:
        image = (image + 1.) / 2.
        image = torch.roll(image, 128, -1)

    mask = torch.zeros_like(image)
    mask[:, :, :, 128:] = 1.

    masked_image = (1. - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


@torch.no_grad()
def convsample_ddim_ipt(model, steps, shape, x_prev=None, eta=1.0, roll_dim=-1):
    bs = shape[0]
    if x_prev is None:
        # batch = make_batch(None, '/home/yifan1/Desktop/ngp-jax/data/gravel_800.jpg', device=model.device)
        # encoder_posterior = model.encode_first_stage(batch['image'])

        plms = PLMSSampler(model)
        bs = shape[0]
        shape = shape[1:]
        samples, intermediates = plms.sample(steps, batch_size=bs, shape=shape, eta=0., verbose=False,)
        return samples, intermediates
    else:
        x = x_prev.detach()
        x = torch.clamp(x, -1., 1.)
        x = torch.roll(x, x.shape[roll_dim] // 2, dims=roll_dim)
        encoder_posterior = model.encode_first_stage(x)

    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = z.tile((bs, 1, 1, 1))
    # z = torch.roll(z, z.shape[roll_dim] // 2, dims=roll_dim)

    b, h, w = z.shape[0], z.shape[2], z.shape[3]
    mask = torch.ones(b, h, w).to(model.device)
    # zeros will be filled in
    if roll_dim == -1:
        mask[:, :, w//2:] = 0.
    elif roll_dim == -2:
        mask[:, h//2:, :] = 0.
    mask = mask[:, None, ...]

    # ddim = DDIMSampler(model)
    # bs = shape[0]
    # shape = shape[1:]
    # samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False,
    #                                      x0=z, mask=mask, )

    plms = PLMSSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = plms.sample(steps, batch_size=bs, shape=shape, eta=0., verbose=False,
                                         x0=z, mask=mask, )

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, x_prev=None, vanilla=False, custom_steps=None, eta=1.0, roll_dim=-1,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        sample, intermediates = convsample_ddim_ipt(model, steps=custom_steps, shape=shape,
                                                    x_prev=x_prev, eta=eta, roll_dim=roll_dim)
        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))
    logs = {'sample': None}

    n_w = 7
    n_samples = n_w ** 2
    img_tile = torch.zeros((3, 256 + 128 * (n_w - 1), 256 + 128 * (n_w - 1)), dtype=torch.float32, device=model.device)
    batch_size = 1

    for i in range(n_w):
        for j in range(n_w):
            # if j > 0:
            #     continue
            # if i == 1 and j == 0:
            #     img_tile_vis = custom_to_pil(img_tile[:, i*128-128:i*128+128, j*128:j*128+256])
            #     imgpath = os.path.join(logdir, f"test_{i}_{j}.png")
            #     img_tile_vis.save(imgpath)

            if i == 0 and j == 0:
                logs = make_convolutional_sample(model, batch_size=batch_size, x_prev=None,
                                                 vanilla=vanilla, custom_steps=custom_steps,
                                                 eta=eta)
            elif i == 0:
                logs = make_convolutional_sample(model, batch_size=batch_size, x_prev=logs['sample'],
                                                 vanilla=vanilla, custom_steps=custom_steps, roll_dim=-1,
                                                 eta=eta)
            elif i > 0:
                logs = make_convolutional_sample(model, batch_size=batch_size,
                                                 x_prev=img_tile[None, :, i*128-128:i*128+128, j*128:j*128+256].clone(),
                                                 vanilla=vanilla, custom_steps=custom_steps, roll_dim=-2,
                                                 eta=eta)
            img_tile[:, i*128:i*128+256, j*128:j*128+256] = logs['sample'][0].clone().detach()
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")

    img_tile_vis = custom_to_pil(img_tile)
    imgpath = os.path.join(logdir, f"tile.png")
    img_tile_vis.save(imgpath)



    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")
