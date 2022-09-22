import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


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
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    config = OmegaConf.load("models/ldm/inpainting_tex/config.yaml")
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load("/home/yifan1/Desktop/latent-diffusion/logs/2022-09-14T15-30-04_tex-ldm-vq-f4-noattn/checkpoints/last.ckpt")["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    n_img = 5
    img_prev = None
    os.makedirs(opt.outdir, exist_ok=True)
    img_out = np.zeros((256 + 128 * (n_img - 1), 256 + 128 * (n_img - 1), 3), dtype=np.uint8)
    with torch.no_grad():
        with model.ema_scope():
            for j in tqdm(range(n_img)):
                for i in range(n_img):
                    outpath = os.path.join(opt.outdir, '{:02d}.png'.format(j * n_img + i))
                    if j == 0:
                        batch = make_batch(img_prev, opt.img_path, device=device)
                    else:
                        img_up = img_out[j*128:j*128+256, i*128:i*128+256, :].copy()
                        batch = make_batch_v(img_up, device=device)

                    # encode masked image and concat downsampled mask
                    c = model.cond_stage_model.encode(batch["masked_image"])
                    cc = torch.nn.functional.interpolate(batch["mask"],
                                                         size=c.shape[-2:])
                    c = torch.cat((c, cc[:, :1, ...]), dim=1)

                    # uc = model.get_learned_conditioning(batch["masked_image"])
                    # uc = torch.cat((uc, cc[:, :1, ...]), dim=1)

                    uc = model.get_learned_conditioning(torch.zeros_like(batch["image"]))
                    uc = torch.cat((uc, torch.ones_like(cc[:, :1, ...])), dim=1)

                    shape = (c.shape[1]-1,)+c.shape[2:]
                    samples_ddim, _ = sampler.sample(S=opt.steps,
                                                     conditioning=c,
                                                     batch_size=c.shape[0],
                                                     shape=shape,
                                                     unconditional_conditioning=uc,
                                                     unconditional_guidance_scale=1.1,
                                                     verbose=False)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    img_prev = x_samples_ddim.detach().clone()

                    image = torch.clamp((batch["image"]+1.0)/2.0,
                                        min=0.0, max=1.0)
                    mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                       min=0.0, max=1.0)
                    predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                  min=0.0, max=1.0)

                    inpainted = (1-mask)*image+mask*predicted_image
                    inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                    Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
                    img_out[j*128:j*128+256, i*128:i*128+256, :] = inpainted
            Image.fromarray(img_out).save(os.path.join(opt.outdir, 'tile.png'))
