import types
from typing import List, Optional


class AmazonStabilityConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0

    Supported Params for the Amazon / Stable Diffusion models:

    - `cfg_scale` (integer): Default `7`. Between [ 0 .. 35 ]. How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)

    - `seed` (float): Default: `0`. Between [ 0 .. 4294967295 ]. Random noise seed (omit this option or use 0 for a random seed)

    - `steps` (array of strings): Default `30`. Between [ 10 .. 50 ]. Number of diffusion steps to run.

    - `width` (integer): Default: `512`. multiple of 64 >= 128. Width of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536

    - `height` (integer): Default: `512`. multiple of 64 >= 128. Height of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536
    """

    cfg_scale: Optional[int] = None
    seed: Optional[float] = None
    steps: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }
