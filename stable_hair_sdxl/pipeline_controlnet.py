from diffusers import StableDiffusionXLControlNetPipeline


class StableHairSDXLControlNetPipeline(StableDiffusionXLControlNetPipeline):
    def prepare_image(self, image, **kwargs):
        image = super().prepare_image(image, **kwargs)
        # norm
        image = 2.0 * image - 1.0
        image = self.vae.encode(image).latent_dist.sample()
        image = image * self.vae.config.scaling_factor
        # the image is downsized by `prepare_image`.
        self.vae_scale_factor = 1
        return image
