import copy
import vision_aided_loss
import torch
import torch.nn as nn
from torch.nn import init
from peft import LoraConfig
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution


def weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def make_1step_sched(model_path):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def initialize_scheduler(model_path="stabilityai/sd-turbo"):
    noise_scheduler_1step = make_1step_sched(model_path)
    return noise_scheduler_1step


def initialize_text_encoder(model_path="stabilityai/sd-turbo"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder").cuda()
    return tokenizer, text_encoder


def initialize_unet(rank, model_path="stabilityai/sd-turbo", return_lora_module_names=False, init_weight=False):
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    if init_weight:
        unet.apply(weights_init)
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out",
              "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]

    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder,
                                   lora_alpha=rank)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder,
                                   lora_alpha=rank)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_modules_others,
                                  lora_alpha=rank)

    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    unet.set_adapters(["default_encoder", "default_decoder", "default_others"])

    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks

    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


def initialize_vae(rank=4, model_path="stabilityai/sd-turbo", return_lora_module_names=False, init_weight=False):
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    if init_weight:
        vae.apply(weights_init)
    vae.requires_grad_(False)
    vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
    vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
    vae.requires_grad_(True)
    vae.train()

    # add the skip connection convs
    vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)\
        .cuda().requires_grad_(True)
    vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).\
        cuda().requires_grad_(True)
    vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).\
        cuda().requires_grad_(True)
    vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).\
        cuda().requires_grad_(True)
    torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
    torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)
    vae.decoder.ignore_skip = False
    vae.decoder.gamma = 1

    l_vae_target_modules = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "skip_conv_1",
                            "skip_conv_2", "skip_conv_3", "skip_conv_4", "to_k", "to_q", "to_v", "to_out.0"]
    vae_lora_config = LoraConfig(r=rank, init_lora_weights="gaussian", target_modules=l_vae_target_modules)
    vae.add_adapter(vae_lora_config, adapter_name="vae_skip")

    if return_lora_module_names:
        return vae, l_vae_target_modules
    else:
        return vae


def initialize_discriminator(gan_loss_type="multilevel_sigmoid", cv_type='conch'):
    net_disc = vision_aided_loss.Discriminator(cv_type=cv_type, loss_type=gan_loss_type, device="cuda")
    net_disc.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False
    return net_disc


class VAE_encode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction="a2b"):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        return _vae.encode(x).latent_dist.sample() * _vae.config.scaling_factor


class VAE_encode_multiview(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_encode_multiview, self).__init__()
        self.vae = vae
        self.vae_patch = copy.deepcopy(vae)
        self.vae_b2a = vae_b2a
        self.vae_b2a_patch = copy.deepcopy(vae_b2a)

    def forward(self, x, direction="a2b", patch_num=4):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
            _vae_patch = self.vae_patch
        else:
            _vae = self.vae_b2a
            _vae_patch = self.vae_b2a_patch

        h1 = _vae.encoder(x)
        _, _c1, _h1, _w1 = x.shape
        _, _c2, _h2, _w2 = h1.shape

        x_patch = x.unfold(2, _h1//patch_num, _w1//patch_num).unfold(3, _h1//patch_num, _w1//patch_num)
        x_patch = x_patch.permute(0, 2, 3, 1, 4, 5).reshape(-1, _c1, _h1//patch_num, _w1//patch_num)

        h2 = _vae_patch.encoder(x_patch)
        h2 = h2.view(patch_num, patch_num, _c2, _h2//patch_num, _w2//patch_num).permute(2, 0, 3, 1, 4)
        h2 = h2.contiguous().view(_c2, _h2, _w2).unsqueeze(0)

        h = h1 + h2

        moments = _vae.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        return posterior.sample() * _vae.config.scaling_factor


class VAE_decode(nn.Module):
    def __init__(self, vae, vae_b2a=None):
        super(VAE_decode, self).__init__()
        self.vae = vae
        self.vae_b2a = vae_b2a

    def forward(self, x, direction="a2b"):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
        else:
            _vae = self.vae_b2a
        assert _vae.encoder.current_down_blocks is not None
        _vae.decoder.incoming_skip_acts = _vae.encoder.current_down_blocks
        x_decoded = _vae.decode(x / _vae.config.scaling_factor).sample.clamp(-1, 1)
        return x_decoded


class VAE_decode_multiview(nn.Module):
    def __init__(self, vae_enc):
        super(VAE_decode_multiview, self).__init__()
        self.vae = vae_enc.vae
        self.vae_b2a = vae_enc.vae_b2a
        self.vae_patch = vae_enc.vae_patch
        self.vae_b2a_patch = vae_enc.vae_b2a_patch

    def forward(self, x, direction="a2b", patch_num=4):
        assert direction in ["a2b", "b2a"]
        if direction == "a2b":
            _vae = self.vae
            _vae_patch = self.vae_patch
        else:
            _vae = self.vae_b2a
            _vae_patch = self.vae_b2a_patch

        assert _vae.encoder.current_down_blocks is not None
        assert _vae_patch.encoder.current_down_blocks is not None

        skip = []
        for h1, h2 in zip(_vae.encoder.current_down_blocks, _vae_patch.encoder.current_down_blocks):
            _, c, h, w = h1.shape
            h2 = h2.view(patch_num, patch_num, c, h//patch_num, w//patch_num).permute(2, 0, 3, 1, 4)
            h2 = h2.contiguous().view(c, h, w).unsqueeze(0)
            skip.append(h1+h2)

        _vae.decoder.incoming_skip_acts = skip
        x_decoded = _vae.decode(x / _vae.config.scaling_factor).sample.clamp(-1, 1)
        return x_decoded


class Diffusion_FFPE(nn.Module):
    def __init__(self, pretrained_path=None, model_path="stabilityai/sd-turbo", lora_rank_unet=128, lora_rank_vae=4,
                 gradient_checkpointing=False, enable_xformers_memory_efficient_attention=False, multi_view=True):

        super().__init__()

        pretrained_sd = None
        if pretrained_path:
            pretrained_sd = torch.load(pretrained_path)
            lora_rank_unet = pretrained_sd['rank_unet']
            lora_rank_vae = pretrained_sd['rank_vae']

        self.sched = initialize_scheduler(model_path)
        self.timesteps = self.sched.config.num_train_timesteps - 1

        vae_a2b = initialize_vae(lora_rank_vae, model_path)
        vae_b2a = copy.deepcopy(vae_a2b)

        if multi_view:
            self.vae_enc = VAE_encode_multiview(vae_a2b, vae_b2a=vae_b2a)
            self.vae_dec = VAE_decode_multiview(self.vae_enc)
        else:
            self.vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
            self.vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

        self.unet = initialize_unet(lora_rank_unet, model_path)

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        if enable_xformers_memory_efficient_attention:
            self.unet.enable_xformers_memory_efficient_attention()

        if pretrained_sd:
            self.load_ckpt_from_state_dict(pretrained_sd)

    def get_trainable_params(self):
        # add all unet parameters
        params_gen = list(self.unet.conv_in.parameters())
        self.unet.conv_in.requires_grad_(True)

        for n, p in self.unet.named_parameters():
            if "lora" in n and "default" in n and 'conv_in' not in n:
                assert p.requires_grad
                params_gen.append(p)

        # add all vae_a2b parameters
        vae_a2b = self.vae_enc.vae
        for n, p in vae_a2b.named_parameters():
            if "lora" in n and "vae_skip" in n and 'decoder.skip_conv' not in n:
                assert p.requires_grad
                params_gen.append(p)

        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_a2b.decoder.skip_conv_4.parameters())

        # add all vae_b2a parameters
        vae_b2a = self.vae_enc.vae_b2a
        for n, p in vae_b2a.named_parameters():
            if "lora" in n and "vae_skip" in n and 'decoder.skip_conv' not in n:
                assert p.requires_grad
                params_gen.append(p)
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_1.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_2.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_3.parameters())
        params_gen = params_gen + list(vae_b2a.decoder.skip_conv_4.parameters())

        return params_gen

    def load_ckpt_from_state_dict(self, sd):
        self.unet.conv_in.load_state_dict(sd['unet_conv_in'], strict=True)
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_encoder.weight", ".weight")
            if "lora" in n and "default_encoder" in n:
                p.data.copy_(sd["sd_encoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_decoder.weight", ".weight")
            if "lora" in n and "default_decoder" in n:
                p.data.copy_(sd["sd_decoder"][name_sd])
        for n, p in self.unet.named_parameters():
            name_sd = n.replace(".default_others.weight", ".weight")
            if "lora" in n and "default_others" in n:
                p.data.copy_(sd["sd_other"][name_sd])

        self.vae_enc.load_state_dict(sd["sd_vae_enc"], strict=True)
        self.vae_dec.load_state_dict(sd["sd_vae_dec"], strict=True)

    def forward(self, x, direction, text_emb):
        assert direction in ["a2b", "b2a"]
        batch_size = x.shape[0]

        timesteps = torch.tensor([self.timesteps] * batch_size, device=x.device).long()
        x_enc = self.vae_enc(x, direction=direction).to(x.dtype)
        model_pred = self.unet(x_enc, timesteps, encoder_hidden_states=text_emb).sample
        x_out = torch.stack([self.sched.step(model_pred[i], timesteps[i], x_enc[i], return_dict=True).prev_sample
                             for i in range(batch_size)])
        x_out_decoded = self.vae_dec(x_out, direction=direction)
        return x_out_decoded
