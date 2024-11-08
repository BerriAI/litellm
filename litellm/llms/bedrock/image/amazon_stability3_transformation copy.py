import types
from typing import List, Optional


class AmazonStability3Config:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0

    Stability API Ref: https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1stable-image~1generate~1sd3/post
    """

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

    def _is_stability_3_model(self, model: str) -> bool:
        """
        Returns True if the model is a Stability 3 model

        Stability 3 models follow this pattern:
            sd3-large
            sd3-large-turbo
            sd3-medium
            sd3.5-large
            sd3.5-large-turbo
        """
        if "sd.3" in model:
            return True
        return False
