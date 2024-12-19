"""
Translates from OpenAI's `/v1/chat/completions` to the VLLM sdk `llm.generate`. 

NOT RECOMMENDED FOR PRODUCTION USE. Use `hosted_vllm/` instead.
"""

from typing import List, Optional, Dict, Any, Union
import types

from litellm.types.llms.openai import AllMessageValues

from ...hosted_vllm.chat.transformation import HostedVLLMChatConfig


class VLLMConfig(HostedVLLMChatConfig):
    """
    VLLM SDK supports the same OpenAI params as hosted_vllm.
    """
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = False
    allowed_local_media_path: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    load_format: str = "auto"
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: Optional[bool] = None
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    hf_overrides: Optional[Any] = None
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    task: str = "auto"
    override_pooler_config: Optional[Any] = None
    compilation_config: Optional[Union[int, Dict[str, Any]]] = None
    
    def __init__(
        self,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[Any] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        task: str = "auto",
        override_pooler_config: Optional[Any] = None,
        compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
    ):
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self":
                setattr(self.__class__, key, value)
                
    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
        }
