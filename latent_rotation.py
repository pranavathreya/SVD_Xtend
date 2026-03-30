import random
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


LATENT_ROTATION_SHIFT_RATIO = 0.25


def roll_horizontally(tensor: torch.Tensor, shift: int) -> torch.Tensor:
    return torch.roll(tensor, shifts=shift, dims=-1)


def random_horizontal_roll(tensor: torch.Tensor) -> torch.Tensor:
    shift = random.randrange(tensor.shape[-1])
    return roll_horizontally(tensor, shift)


def make_latent_rotation_callback(shift_ratio: float = LATENT_ROTATION_SHIFT_RATIO):
    def callback_on_step_end(_pipe, _step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        shift = max(1, int(latents.shape[-1] * shift_ratio))
        callback_kwargs["latents"] = roll_horizontally(latents, shift)
        return callback_kwargs

    return callback_on_step_end


def _circular_conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias):
    working = F.pad(input, self.paddingX, mode=self.padding_modeX)
    working = F.pad(working, self.paddingY, mode=self.padding_modeY)
    return F.conv2d(working, weight, bias, self.stride, _pair(0), self.dilation, self.groups)


@contextmanager
def circular_panorama_padding(unet, vae, tile_x: bool = True, tile_y: bool = False):
    patched_modules = []
    modules_all = [item for item in unet.named_modules()] + [item for item in vae.named_modules()]

    for _, module in modules_all:
        if isinstance(module, torch.nn.Conv2d):
            module.padding_modeX = "circular" if tile_x else "constant"
            module.padding_modeY = "circular" if tile_y else "constant"
            module.paddingX = (
                module._reversed_padding_repeated_twice[0],
                module._reversed_padding_repeated_twice[1],
                0,
                0,
            )
            module.paddingY = (
                0,
                0,
                module._reversed_padding_repeated_twice[2],
                module._reversed_padding_repeated_twice[3],
            )
            module._original_conv_forward = module._conv_forward
            module._conv_forward = _circular_conv_forward.__get__(module, torch.nn.Conv2d)
            patched_modules.append(module)

    try:
        yield
    finally:
        for module in patched_modules:
            module._conv_forward = module._original_conv_forward
            del module._original_conv_forward
