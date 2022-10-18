import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
import k_diffusion as K
import accelerate
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


class CFGDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def make_batch(image, img_path, device):
    if image is None:
        image = np.array(Image.open(img_path).convert("RGB"))
        image = image.astype(np.float32)/255.0
        image = image[None].transpose(0,3,1,2)
        image = torch.from_numpy(image)
        # image = image[:, :, -356:-100, -356:-100]
        image = image[:, :, :256, :256]

        mask = torch.zeros_like(image)

        mask = torch.ones_like(image)
        masked_image = (1. - mask) * image
        #
        # batch = {"image": image, "mask": mask, "masked_image": masked_image}
        # for k in batch:
        #     batch[k] = batch[k].to(device=device)
        #     batch[k] = batch[k] * 2.0 - 1.0
        # return batch
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

def make_batch_v(image, device):
    image = image.astype(np.float32) / 255.
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = torch.zeros_like(image)
    mask[:, :, 128:, :] = 1.

    masked_image = (1. - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch

def make_batch_q(image, device):
    image = image.astype(np.float32) / 255.
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = torch.zeros_like(image)
    mask[:, :, 128:, 128:] = 1.

    masked_image = (1. - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        type=str,
        nargs="?",
        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    opt.outdir = '/home/yifan1/Desktop/latent-diffusion/outputs/rock_7_quad_rot_lms'
    # config_path = '/home/yifan1/Desktop/latent-diffusion/models/ldm/inpainting_tex/tex-ldm-vq-f4-ipt.yaml'
    # ckpt_path = '/home/yifan1/Desktop/latent-diffusion/logs/2022-10-03T17-12-41_tex-ldm-vq-f4-ipt/checkpoints/last.ckpt'

    # config_path = '/home/yifan1/Desktop/latent-diffusion/models/ldm/inpainting_tex/tex-ldm-vq-f4-ipt.yaml'
    # ckpt_path = '/home/yifan1/Desktop/latent-diffusion/logs/2022-10-05T21-16-45_tex-ldm-vq-f4-ipt/checkpoints/last.ckpt'

    config_path = '/home/yifan1/Desktop/latent-diffusion/models/ldm/inpainting_tex/tex-ldm-vq-f8-ipt.yaml'
    ckpt_path = '/home/yifan1/Desktop/latent-diffusion/logs/2022-10-07T20-27-00_tex-ldm-vq-f8-ipt/checkpoints/last.ckpt'

    # config_path = '/home/yifan1/Desktop/latent-diffusion/models/ldm/inpainting_tex/tex-ldm-vq-f4-n256-ipt.yaml'
    # ckpt_path = '/home/yifan1/Desktop/latent-diffusion/logs/2022-10-11T18-01-48_tex-ldm-vq-f4-n256-ipt/checkpoints/last.ckpt'

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    # sampler = DDIMSampler(model)
    # sampler = PLMSSampler(model)

    model_wrap = K.external.CompVisDenoiser(model)
    sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

    n_img = 7
    n_saved = 0
    uc_scale = 2.25
    opt.outdir = f'{opt.outdir}_{uc_scale}'
    print(opt.outdir)
    accelerator = accelerate.Accelerator()

    img_prev = None
    os.makedirs(opt.outdir, exist_ok=True)
    img_out = np.zeros((256 + 128 * (n_img - 1), 256 + 128 * (n_img - 1), 3), dtype=np.uint8)
    with torch.no_grad():
        with model.ema_scope():
            for i in range(n_img):
                outpath = os.path.join(opt.outdir, '{:02d}.png'.format(n_saved))
                n_saved += 1
                batch = make_batch(img_prev, opt.img_path, device=device)


                # encode masked image and concat downsampled mask
                # c = model.cond_stage_model.encode(batch["masked_image"])
                c = model.get_learned_conditioning(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc[:, :1, ...]), dim=1)

                uc = model.get_learned_conditioning(torch.zeros_like(batch["image"]).fill_(-1.))
                uc = torch.cat((uc, torch.ones_like(cc[:, :1, ...])), dim=1)

                shape = (c.shape[1] - 1,) + c.shape[2:]

                # x0 = model.get_first_stage_encoding(model.encode_first_stage(batch["masked_image"]))  # move to latent space
                # x0 = torch.randn(shape, device=device).unsqueeze(0)
                # sigmas = model_wrap.get_sigmas(opt.steps)
                # torch.manual_seed(23)  # changes manual seeding procedure
                # noise = torch.randn_like(x0) * sigmas[opt.steps - t_enc - 1]  # for GPU draw
                # xi = x0 + noise
                # sigma_sched = sigmas[opt.steps - t_enc - 1:]

                sigmas = model_wrap.get_sigmas(opt.steps)
                xi = torch.randn(shape, device=device).unsqueeze(0) * sigmas[0]

                model_wrap_cfg = CFGDenoiser(model_wrap)
                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': uc_scale}
                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigmas, extra_args=extra_args,
                                                     disable=not accelerator.is_main_process)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                img_prev = x_samples_ddim.detach().clone()

                image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                              min=0.0, max=1.0)

                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                img_out[:256, i * 128:i * 128 + 256, :] = inpainted
            Image.fromarray(img_out).save(os.path.join(opt.outdir, 'tile_w.png'))

            for i in range(1, n_img):
                outpath = os.path.join(opt.outdir, '{:02d}.png'.format(n_saved))
                n_saved += 1
                img_up = img_out[i * 128:i * 128 + 256, :256, :].copy()
                batch = make_batch_v(img_up, device=device)

                # encode masked image and concat downsampled mask
                # c = model.cond_stage_model.encode(batch["masked_image"])
                c = model.get_learned_conditioning(batch["masked_image"])
                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc[:, :1, ...]), dim=1)

                uc = model.get_learned_conditioning(torch.zeros_like(batch["image"]).fill_(-1.))
                uc = torch.cat((uc, torch.ones_like(cc[:, :1, ...])), dim=1)

                shape = (c.shape[1] - 1,) + c.shape[2:]

                sigmas = model_wrap.get_sigmas(opt.steps)
                xi = torch.randn(shape, device=device).unsqueeze(0) * sigmas[0]

                model_wrap_cfg = CFGDenoiser(model_wrap)
                extra_args = {'cond': c, 'uncond': uc, 'cond_scale': uc_scale}
                samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigmas, extra_args=extra_args,
                                                     disable=not accelerator.is_main_process)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                img_prev = x_samples_ddim.detach().clone()

                image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                              min=0.0, max=1.0)

                inpainted = (1 - mask) * image + mask * predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                img_out[i * 128:i * 128 + 256, :256, :] = inpainted
            Image.fromarray(img_out).save(os.path.join(opt.outdir, 'tile_h.png'))

            for i in range(n_img - 1):
                for j in range(n_img - 1):
                    outpath = os.path.join(opt.outdir, '{:02d}.png'.format(n_saved))
                    n_saved += 1
                    img_up = img_out[(i + 1) * 128:(i + 3) * 128, (j + 1) * 128:(j + 3) * 128, :].copy()
                    batch = make_batch_q(img_up, device=device)

                    # encode masked image and concat downsampled mask
                    # c = model.cond_stage_model.encode(batch["masked_image"])
                    c = model.get_learned_conditioning(batch["masked_image"])
                    cc = torch.nn.functional.interpolate(batch["mask"],
                                                         size=c.shape[-2:])
                    c = torch.cat((c, cc[:, :1, ...]), dim=1)

                    uc = model.get_learned_conditioning(torch.zeros_like(batch["image"]).fill_(-1.))
                    uc = torch.cat((uc, torch.ones_like(cc[:, :1, ...])), dim=1)

                    shape = (c.shape[1] - 1,) + c.shape[2:]
                    sigmas = model_wrap.get_sigmas(opt.steps)
                    xi = torch.randn(shape, device=device).unsqueeze(0) * sigmas[0]

                    model_wrap_cfg = CFGDenoiser(model_wrap)
                    extra_args = {'cond': c, 'uncond': uc, 'cond_scale': uc_scale}
                    samples_ddim = K.sampling.sample_lms(model_wrap_cfg, xi, sigmas, extra_args=extra_args,
                                                         disable=not accelerator.is_main_process)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    img_prev = x_samples_ddim.detach().clone()

                    image = torch.clamp((batch["image"] + 1.0) / 2.0,
                                        min=0.0, max=1.0)
                    mask = torch.clamp((batch["mask"] + 1.0) / 2.0,
                                       min=0.0, max=1.0)
                    predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                  min=0.0, max=1.0)

                    inpainted = (1 - mask) * image + mask * predicted_image
                    inpainted = inpainted.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
                    Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                    img_out[(i + 2) * 128:(i + 1) * 128 + 256, (j + 2) * 128:(j + 1) * 128 + 256, :] = \
                        inpainted[128:, 128:, :]
            Image.fromarray(img_out).save(os.path.join(opt.outdir, 'tile.png'))