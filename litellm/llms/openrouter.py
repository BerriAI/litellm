from typing import List, Dict
import types


class OpenrouterConfig:
    """
    Reference: https://openrouter.ai/docs#format

    """

    # OpenRouter-only parameters
    extra_body: Dict[str, List[str]] = {"transforms": []}  # default transforms to []

    def __init__(
        self,
        transforms: List[str] = [],
        models: List[str] = [],
        route: str = "",
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
