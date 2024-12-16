import copy

from diffusers import ControlNetModel


class StableHairControlNetModel(ControlNetModel):
    def __init__(
        self,
        in_channels: int = 4,
        conditioning_channels: int = 4,
        block_out_channels: tuple[int, ...] = (320, 640, 1280, 1280),
        **kwargs,
    ) -> None:
        super().init(
            in_channels=in_channels,
            conditioning_channels=conditioning_channels,
            **kwargs,
        )
        self.controlnet_cond_embedding = copy.deepcopy(self.conv_in)
