import types
from typing import Final


class PalmConfig:
    """
    Reference: https://developers.generativeai.google/api/python/google/generativeai/chat

    The class `PalmConfig` provides configuration for the Palm's API interface. Here are the parameters:

    - `context` (string): Text that should be provided to the model first, to ground the response. This could be a prompt to guide the model's responses.

    - `examples` (list): Examples of what the model should generate. They are treated identically to conversation messages except that they take precedence over the history in messages if the total input size exceeds the model's input_token_limit.

    - `temperature` (float): Controls the randomness of the output. Must be positive. Higher values produce a more random and varied response. A temperature of zero will be deterministic.

    - `candidate_count` (int): Maximum number of generated response messages to return. This value must be between [1, 8], inclusive. Only unique candidates are returned.

    - `top_k` (int): The API uses combined nucleus and top-k sampling. `top_k` sets the maximum number of tokens to sample from on each step.

    - `top_p` (float): The API uses combined nucleus and top-k sampling. `top_p` configures the nucleus sampling. It sets the maximum cumulative probability of tokens to sample from.

    - `max_output_tokens` (int): Sets the maximum number of tokens to be returned in the output
    """

    context: str | None = None
    examples: list | None = None
    temperature: float | None = None
    candidate_count: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None

    def __init__(
        self,
        context: str | None = None,
        examples: list | None = None,
        temperature: float | None = None,
        candidate_count: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        locals_: Final = locals().copy()
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
