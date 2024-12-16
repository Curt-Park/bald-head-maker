from diffusers import StableDiffusionXLControlNetPipeline


class StableHairSDXLControlNetPipeline(StableDiffusionXLControlNetPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vae_scale_factor = 1

    def prepare_image(self, image, **kwargs):
        image = super().prepare_image(image, **kwargs)
        # norm
        image = 2.0 * image - 1.0
        image = self.vae.encode(image).latent_dist.sample()
        image = image * self.vae.config.scaling_factor
        return image
