"""
Nvidia NIM embeddings endpoint: https://docs.api.nvidia.com/nim/reference/nvidia-nv-embedqa-e5-v5-infer

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

import types
from typing import Optional, Union


class NvidiaNimEmbeddingConfig:
    """
    Reference: https://docs.api.nvidia.com/nim/reference/nvidia-nv-embedqa-e5-v5-infer
    """

    # OpenAI params
    encoding_format: Optional[str] = None
    user: Optional[str] = None

    # Nvidia NIM params
    input_type: Optional[str] = None
    truncate: Optional[str] = None

    def __init__(
        self,
        encoding_format: Optional[str] = None,
        user: Optional[str] = None,
        input_type: Optional[str] = None,
        truncate: Optional[str] = None,
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

    def get_supported_openai_params(
        self,
    ):
        return ["encoding_format", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        kwargs: Optional[dict] = None,
    ):
        if "extra_body" not in optional_params:
            optional_params["extra_body"] = {}
        for k, v in non_default_params.items():
            if k == "input_type":
                optional_params["extra_body"].update({"input_type": v})
            elif k == "truncate":
                optional_params["extra_body"].update({"truncate": v})

        if kwargs is not None:
            # pass kwargs in extra_body
            optional_params["extra_body"].update(kwargs)
        return optional_params
