from typing import Optional, Union, List, Dict, Any
from PIL import Image
import io
import base64
import time

try:
    from diffusers import StableDiffusionPipeline
except:
    pass
from pydantic import BaseModel


class ImageResponse(BaseModel):
    created: int
    data: List[Dict[str, str]]  # List of dicts with "b64_json" or "url"


class DiffusersImageHandler:
    def __init__(self):
        self.pipeline_cache = {}  # Cache loaded pipelines
        self.device = self._get_default_device()

    def _get_default_device(self):
        """Determine the best available device"""
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():  # For Apple Silicon
            return "mps"
        else:
            return "cpu"

    def _load_pipeline(
        self, model: str, device: Optional[str] = None
    ) -> StableDiffusionPipeline:
        """Load and cache diffusion pipeline"""
        device = device or self.device

        if model not in self.pipeline_cache:
            try:
                pipe = StableDiffusionPipeline.from_pretrained(model)
                pipe = pipe.to(device)
                self.pipeline_cache[model] = pipe
            except RuntimeError as e:
                if "CUDA" in str(e):
                    # Fallback to CPU if CUDA fails
                    verbose_logger.warning(f"Falling back to CPU: {str(e)}")
                    pipe = pipe.to("cpu")
                    self.pipeline_cache[model] = pipe
                else:
                    raise

        return self.pipeline_cache[model]

    def _image_to_b64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def generate_image(
        self,
        prompt: str,
        model: str = "runwayml/stable-diffusion-v1-5",
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        device: str = "cuda",
        **kwargs,
    ) -> ImageResponse:
        """
        Generate image from text prompt
        Args:
            prompt: Text prompt to generate image from
            model: Diffusers model ID
            height: Height of output image
            width: Width of output image
            num_inference_steps: Number of denoising steps
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Negative prompt to avoid certain content
            num_images_per_prompt: Number of images to generate
            device: Device to run on ('cuda' or 'cpu')
        Returns:
            ImageResponse with base64 encoded images
        """
        pipe = self._load_pipeline(model, device)

        # Generate image(s)
        images = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            **kwargs,
        ).images

        # Convert to response format
        image_data = [{"b64_json": self._image_to_b64(img)} for img in images]

        return ImageResponse(created=int(time.time()), data=image_data)

    def generate_variation(
        self,
        image: Union[Image.Image, str, bytes],  # Accepts PIL, file path, or bytes
        prompt: Optional[str] = None,
        model: str = "runwayml/stable-diffusion-v1-5",
        strength: float = 0.8,
        **kwargs,
    ) -> ImageResponse:
        """
        Generate variation of input image
        Args:
            image: Input image (PIL, file path, or bytes)
            prompt: Optional text prompt to guide variation
            model: Diffusers model ID
            strength: Strength of variation (0-1)
        Returns:
            ImageResponse with base64 encoded images
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))

        pipe = self._load_pipeline(model)

        # Generate variation
        result = pipe(prompt=prompt, image=image, strength=strength, **kwargs)

        # Convert to response format
        image_data = [{"b64_json": self._image_to_b64(result.images[0])}]

        return ImageResponse(created=int(time.time()), data=image_data)
