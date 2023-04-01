import torch.nn as nn
from diffusers import DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from controlnet_aux import HEDdetector
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
from diffusers.utils import load_image

from sd import SpecifyGradient, seed_everything


# condition type to (controlnet model, stable-diffusion model)
type2model_id = {
    "scribble": ("fusing/stable-diffusion-v1-5-controlnet-scribble", "runwayml/stable-diffusion-v1-5")
}


class ControlNet(nn.Module):
    def __init__(self, image_path, device, fp16, vram_O, type='scribble', hed=False):
        """

        Args:
            device:
            image_path:
            fp16:
            vram_O:
            type:
        """
        super().__init__()

        self.device = device

        print(f'[INFO] loading controlnet...')

        self.image = load_image(image_path)
        if hed:
            hed = HEDdetector.from_pretrained('lllyasviel/ControlNet')
            self.image = hed(self.image, scribble=True)
            # TODO call self.pipe.prepare_image

        controlnet_model_id, sd_model_id = type2model_id[type]
        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)

        precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            sd_model_id, controlnet=controlnet, safety_checker=None, torch_dtype=precision_t,
            # controlnet_conditioning_scale=1.0
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Remove if you do not have xformers installed
        # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
        # for installation instructions
        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            if is_xformers_available():
                pipe.enable_xformers_memory_efficient_attention()
            pipe.to(device)

        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.controlnet = pipe.controlnet

        self.scheduler = DDIMScheduler.from_pretrained(sd_model_id, subfolder="scheduler", torch_dtype=precision_t)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        print(f'[INFO] loaded control net!')

    def get_text_embeds(self, prompt, negative_prompt):
        """ Same as StableDiffusion.get_text_embeds
        """
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt'
        )

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        latents = self.encode_imgs(pred_rgb_512)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # TODO: add self.controlnet in here
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # Save input tensors for UNet
            #torch.save(latent_model_input, "train_latent_model_input.pt")
            #torch.save(t, "train_t.pt")
            #torch.save(text_embeddings, "train_text_embeddings.pt")
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings
            ).sample

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # since we omitted an item in grad, we need to use the custom function to specify the gradient
        loss = SpecifyGradient.apply(latents, grad)

        return loss

    def produce_latents(
            self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None
    ):
        if latents is None:
            latents = torch.randn(
                (text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device
            )

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # Save input tensors for UNet
                #torch.save(latent_model_input, "produce_latents_latent_model_input.pt")
                #torch.save(t, "produce_latents_t.pt")
                #torch.save(text_embeddings, "produce_latents_text_embeddings.pt")
                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    def decode_latents(self, latents):
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215
        return latents

    def prompt_to_img(
            self, prompts, negative_prompts='', height=512, width=512,
            num_inference_steps=50, guidance_scale=7.5, latents=None
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(
            text_embeds, height=height, width=width, latents=latents,
            num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)  # [1, 4, 64, 64]

        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')
        return imgs


if __name__ == '__main__':
    """
    """
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('guidance_image_path', type=str, default=None)

    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--controlnet_type', type=str, default='scribble', choices=['scribble'])
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)
    device = torch.device('cuda')

    controlnet = ControlNet(opt.guidance_image_path, device, opt.fp16, opt.vram_O, opt.controlnet_type)
    imgs = controlnet.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
