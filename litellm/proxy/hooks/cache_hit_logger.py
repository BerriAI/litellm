"""
MiMo 缓存命中日志回调 v3
智能提取 usage，兼容流式/非流式/冷启动 edge case
"""

import logging
import json
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.cache_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[CACHE] %(message)s"))
    logger.addHandler(handler)

_raw_dump_count = 0
_MAX_RAW_DUMPS = 3


def _deep_get(obj, *keys):
    """安全嵌套取值，支持对象属性和字典"""
    for key in keys:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            obj = getattr(obj, key, None)
    return obj


def _to_int(val):
    """安全转 int，None/异常返回 0"""
    if val is None:
        return 0
    try:
        return int(val)
    except (ValueError, TypeError):
        return 0


def _parse_usage_dict(u):
    """从 dict 格式的 usage 提取四元组"""
    if not u or not isinstance(u, dict):
        return None
    prompt = _to_int(u.get("prompt_tokens"))
    details = u.get("prompt_tokens_details") or {}
    cached = _to_int(details.get("cached_tokens"))
    creation = _to_int(details.get("cache_creation_tokens"))
    completion = _to_int(u.get("completion_tokens"))
    if prompt > 0:
        return prompt, cached, creation, completion
    return None


def _extract_usage(response_obj, kwargs):
    """
    6 路径智能提取 usage:
    1. response_obj.usage (标准对象属性)
    2. response_obj 是 dict
    3. kwargs 中的 original_response / hidden_params / response
    4. 遍历 kwargs 找含 usage 的 dict
    5. response_obj._hidden_params.stream_usage
    6. response_obj._hidden_params.usage  ← 冷启动时的真实值
    """
    # 路径1: response_obj.usage（标准对象属性）
    usage = _deep_get(response_obj, "usage")
    if usage:
        prompt = _to_int(_deep_get(usage, "prompt_tokens"))
        cached = _to_int(_deep_get(usage, "prompt_tokens_details", "cached_tokens"))
        creation = _to_int(
            _deep_get(usage, "prompt_tokens_details", "cache_creation_tokens")
        )
        completion = _to_int(_deep_get(usage, "completion_tokens"))
        if prompt > 0:
            return prompt, cached, creation, completion

    # 路径2: response_obj 是 dict
    if isinstance(response_obj, dict):
        result = _parse_usage_dict(response_obj.get("usage"))
        if result:
            return result

    # 路径3: kwargs 中的 hidden/original response
    for key in ("original_response", "hidden_params", "response"):
        candidate = kwargs.get(key)
        if candidate is None:
            continue
        u = _deep_get(candidate, "usage")
        if u is None and isinstance(candidate, dict):
            u = candidate.get("usage")
        result = _parse_usage_dict(u) if isinstance(u, dict) else None
        if result:
            return result

    # 路径4: 遍历 kwargs 找含 usage 的 dict
    for k, v in kwargs.items():
        if isinstance(v, dict) and "usage" in v:
            result = _parse_usage_dict(v["usage"])
            if result:
                return result

    # 路径5: _hidden_params.stream_usage
    hidden = getattr(response_obj, "_hidden_params", None)
    if hidden and isinstance(hidden, dict):
        result = _parse_usage_dict(hidden.get("stream_usage"))
        if result:
            return result

        # 路径6: _hidden_params.usage（冷启动时的真实原始账单）
        result = _parse_usage_dict(hidden.get("usage"))
        if result:
            return result

    return None


def _dump_raw(response_obj, kwargs, model):
    """打印原始 usage 结构用于调试（限 3 次）"""
    global _raw_dump_count
    if _raw_dump_count >= _MAX_RAW_DUMPS:
        return
    _raw_dump_count += 1
    try:
        ru = _deep_get(response_obj, "usage")
        ru_str = "None"
        if ru is not None:
            if isinstance(ru, dict):
                ru_str = json.dumps(ru, default=str, ensure_ascii=False)[:800]
            else:
                # Pydantic model - 拿实际字段值
                fields = (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_tokens_details",
                    "completion_tokens_details",
                )
                ru_str = str({k: getattr(ru, k, None) for k in fields})
        logger.info(f"RAW DUMP #{_raw_dump_count} model={model}")
        logger.info(f"  response_obj.usage = {ru_str}")

        # _hidden_params.usage（关键！冷启动账单在这里）
        hp = getattr(response_obj, "_hidden_params", None)
        if hp and isinstance(hp, dict):
            hp_usage = hp.get("usage")
            if hp_usage:
                logger.info(
                    f"  _hidden_params.usage = {json.dumps(hp_usage, default=str, ensure_ascii=False)[:800]}"
                )
            else:
                logger.info(f"  _hidden_params keys = {list(hp.keys())[:20]}")
    except Exception as e:
        logger.error(f"RAW DUMP error: {e}")


class CacheHitLogger(CustomLogger):
    """记录每次 API 调用的缓存命中情况"""

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._log_cache_info(kwargs, response_obj)
        except Exception as e:
            logger.error(f"Error in async_log_success_event: {e}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            self._log_cache_info(kwargs, response_obj)
        except Exception as e:
            logger.error(f"Error in log_success_event: {e}")

    def _log_cache_info(self, kwargs, response_obj):
        model = kwargs.get("model", "unknown")
        completion = _to_int(_deep_get(response_obj, "usage", "completion_tokens"))

        result = _extract_usage(response_obj, kwargs)

        if result is None:
            if completion > 0:
                # 小米冷启动已知 edge case：prompt_tokens=0 是真实返回值
                # 降级为 WARNING，不刷 RAW DUMP
                prompt_raw = _to_int(_deep_get(response_obj, "usage", "prompt_tokens"))
                if prompt_raw == 0:
                    logger.warning(
                        f"COLD_START model={model} | "
                        f"prompt=0(completion={completion}) | "
                        f"小米冷启动未返回 prompt_tokens，跳过"
                    )
                else:
                    _dump_raw(response_obj, kwargs, model)
                    logger.error(
                        f"PARSE_FAIL model={model} | "
                        f"prompt={prompt_raw} completion={completion} | "
                        f"无法提取 usage"
                    )
            return

        prompt, cached, creation, comp = result
        if comp == 0:
            comp = completion

        hit_rate = (cached / prompt * 100) if prompt > 0 else 0.0

        logger.info(
            f"model={model} | "
            f"prompt={prompt} | "
            f"cached_hit={cached} | "
            f"cache_creation={creation} | "
            f"completion={comp} | "
            f"hit_rate={hit_rate:.1f}%"
        )


cache_logger = CacheHitLogger()
