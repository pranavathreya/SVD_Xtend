import random

import torch


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
