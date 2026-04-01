"""
ForceParamsHook - 强制覆盖请求参数

支持两级配置：
1. 全局级别：litellm_settings.force_params（或代码中 litellm.force_params）
2. 模型级别：model_list[].litellm_params.force_params（优先级高于全局）

两级均存在时，模型级别的值会覆盖全局级别中相同 key 的值。
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    pass


class ForceParamsHook(CustomLogger):
    """
    强制覆盖请求参数的 Hook。

    在 async_pre_call_deployment_hook 中执行，此时 deployment 已选定，
    litellm_params 中的 force_params 已合并进 kwargs，可同时获取全局与模型级别配置。

    配置示例（config.yaml）：

        litellm_settings:
          force_params:
            temperature: 0.0
            max_tokens: 2048

        model_list:
          - model_name: my-gpt4
            litellm_params:
              model: openai/gpt-4
              force_params:
                temperature: 0.7   # 覆盖全局 0.0，仅对此模型生效
    """

    async def async_pre_call_deployment_hook(
        self,
        kwargs: Dict[str, Any],
        call_type: Optional[CallTypes],
    ) -> Optional[dict]:
        """
        在 deployment 选定后、请求发出前执行。

        优先级（低 → 高）：用户请求 < 全局 force_params < 模型级 force_params
        """
        # 1. 全局 force_params
        global_force: Dict[str, Any] = {}
        if litellm.force_params and isinstance(litellm.force_params, dict):
            global_force = litellm.force_params

        # 2. 模型级 force_params（来自 litellm_params，由 Router 展开进 kwargs）
        model_force: Dict[str, Any] = {}
        raw = kwargs.get("force_params")
        if raw and isinstance(raw, dict):
            model_force = raw

        # 无任何配置，跳过
        if not global_force and not model_force:
            return None

        # 3. 合并：模型级覆盖全局
        merged: Dict[str, Any] = {**global_force, **model_force}

        verbose_proxy_logger.debug(
            "ForceParamsHook: applying force_params=%s", merged
        )

        kwargs.update(merged)
        return kwargs
