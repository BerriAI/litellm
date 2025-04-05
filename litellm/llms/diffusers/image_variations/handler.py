from typing import Optional, Union
from PIL import Image
import io
import base64
import time

try:
    from diffusers import StableDiffusionPipeline
except:
    pass


class DiffusersImageHandler:
    def __init__(self):
        self.pipeline_cache = {}  # Cache loaded models
        self.device = self._get_default_device()

    def _load_pipeline(self, model: str, device: str = "cuda"):
        """Load and cache diffusion pipeline"""
        if model not in self.pipeline_cache:
            self.pipeline_cache[model] = StableDiffusionPipeline.from_pretrained(
                model
            ).to(device)
        return self.pipeline_cache[model]

    def generate_image(
        self,
        prompt: str,
        model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        **kwargs
    ) -> Image.Image:
        """Generate image from text prompt"""
        pipe = self._load_pipeline(model, device)
        return pipe(prompt, **kwargs).images[0]

    def generate_variation(
        self,
        image: Union[Image.Image, str, bytes],  # Accepts PIL, file path, or bytes
        model: str = "runwayml/stable-diffusion-v1-5",
        device: str = "cuda",
        **kwargs
    ) -> Image.Image:
        """Generate variation of input image"""
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        pipe = self._load_pipeline(model, device)
        return pipe(image=image, **kwargs).images[0]

    def generate_to_bytes(self, *args, **kwargs) -> bytes:
        """Generate image and return as bytes"""
        img = self.generate_image(*args, **kwargs)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

    def generate_to_b64(self, *args, **kwargs) -> str:
        """Generate image and return as base64"""
        return base64.b64encode(self.generate_to_bytes(*args, **kwargs)).decode("utf-8")
