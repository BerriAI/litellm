# LiteLLM Vertex AI 图片响应内存泄漏：分析与修复

## 问题背景

生产环境运行 LiteLLM（`/usr/bin/litellm --use_prisma_db_push --config=...`）时，通过 memray 内存分析发现堆内存持续增长，两次快照均接近 OOM。

---

## Memray 快照对比

### 第一次快照（修复前）
> Duration: 838.7s | Heap: **1.085GB（90% of 1.200GB max）**

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
model_dump_json                     479.441MB    44.20%   479.441MB   44.20%  37   pydantic.main
text                                296.883MB    27.37%   296.883MB   27.37%  22   httpx._models
_extract_image_response_from_parts  173.287MB    15.97%   173.287MB   15.97%  21   litellm.llms.vertex
raw_decode                          124.626MB    11.49%   124.626MB   11.49%  22   json.decoder
_acompletion                          1.074GB    98.96%    19.968kB    0.00%  223  litellm.router
```

### 第二次快照（Fix 1-3 后，Fix 4 前）
> Duration: 503.7s | Heap: **742MB（96% of 772MB max）**

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
model_dump_json                     333.498MB    44.93%   333.498MB   44.93%  28   pydantic.main
text                                258.418MB    34.81%   258.418MB   34.81%  21   httpx._models
raw_decode                          106.960MB    14.41%   106.960MB   14.41%  20   json.decoder
_extract_image_response_from_parts   22.768MB     3.07%    22.768MB    3.07%   3   litellm.llms.vertex
encode_content                       15.115MB     2.04%    15.115MB    2.04%   7   httpx._content
_acompletion                        387.242MB    52.17%     6.144kB    0.00%  94   litellm.router
```

### 修复效果对比

| 函数 | 修复前 | 修复后 | 状态 |
|------|--------|--------|------|
| `_extract_image_response_from_parts` | 173MB | **22MB** | ✅ Fix 2 显著生效 |
| `model_dump_json` | 479MB | 333MB | ✅ Fix 3 降低峰值 |
| `text` (httpx) | 297MB | **258MB** | ❌ 几乎无变化，另有根因 |
| `raw_decode` | 124MB | 107MB | 略降 |

---

## 根因分析

同一份 Vertex AI 图片数据（base64 编码）在处理链路上被**复制了 4 次**，且被多个生命周期较长的对象引用，导致 GC 无法回收。

### 数据复制路径

```
Vertex 返回 JSON（含 inlineData.data base64 图片）
        │
        ├─① httpx.Response.text ──────────────────── 297MB
        │   transform_response() 调用 raw_response.text
        │   → post_call(original_response=raw_response.text)
        │   → litellm.error_logs["POST_CALL"] = locals()  ◄── 全局 dict，永久持有
        │   → model_call_details["original_response"]     ◄── 常驻至 callbacks 完成
        │
        ├─② json.decoder.raw_decode ─────────────── 124MB
        │   将 JSON 字符串解析为 Python dict
        │   包含 part["inlineData"]["data"] 字段
        │
        ├─③ _extract_image_response_from_parts ───── 173MB
        │   data_uri = f"data:{mime_type};base64,{data}"  ← 完整复制 base64
        │   → ImageURLListItem → StreamingChoices.delta.images
        │   → model_call_details["complete_streaming_response"]  ◄── 常驻内存
        │
        └─④ model_dump_json ────────────────────────── 479MB
            caching_handler.py:
            asyncio.create_task(cache.async_add_cache(
                result.model_dump_json(),  ← 提前序列化，479MB 字符串
                ...                          在 task 入队时就已分配
            ))
```

### 各分配点持有者

| 分配点 | 被谁持有 | 释放时机 |
|--------|----------|----------|
| `litellm.error_logs["POST_CALL"]["original_response"]` 297MB | **全局模块 dict** | 下次 `post_call` 调用时才覆盖 |
| `model_call_details["original_response"]` | `Logging` 对象 | 所有 callback 执行完毕后 |
| `complete_streaming_response` 173MB | `model_call_details` | 同上 |
| `model_dump_json()` 序列化字符串 479MB | `asyncio.Task` 引用的协程参数 | task 执行完毕后 |
| `standard_logging_object.response` | `StandardLoggingPayload` → callback | 转发给 Langfuse/DataDog/DB 等 |

**Fix 1 未能解决 `text` 258MB 的真实原因**：

```python
def post_call(self, original_response, ...):
    litellm.error_logs["POST_CALL"] = locals()  # ← 在截断之前！存了完整 258MB
    ...
    # Fix 1 只截断了 model_call_details，但 error_logs 里的全局引用仍在
    self.model_call_details["original_response"] = truncate_base64_in_messages(...)
```

`locals()` 是 Python 内建函数，快照当前帧所有局部变量。`error_logs` 是模块级全局 dict，只有下次 `post_call` 调用时才被覆盖——在高并发场景下，每个请求的完整响应体都会在内存里存活到下次请求。

---

## 修复方案

### Fix 1（已修订）：截断提前至 `locals()` 之前

**文件**：`litellm/litellm_core_utils/litellm_logging.py`

**问题**：`post_call` 第一行 `litellm.error_logs["POST_CALL"] = locals()` 在截断之前执行，全局 dict 持有完整的 258MB 字符串。

```python
# 修复前
def post_call(self, original_response, ...):
    litellm.error_logs["POST_CALL"] = locals()        # ← 存了完整 258MB
    if isinstance(original_response, dict):
        original_response = json.dumps(original_response)
    try:
        self.model_call_details["original_response"] = truncate_base64_in_messages(
            original_response                         # ← 截断只影响 model_call_details
        ) if isinstance(original_response, (str, list, dict)) else original_response

# 修复后
def post_call(self, original_response, ...):
    # 截断提前到 locals() 之前，error_logs 和 model_call_details 均使用截断版本
    if isinstance(original_response, dict):
        original_response = json.dumps(original_response)
    if isinstance(original_response, (str, list, dict)):
        original_response = truncate_base64_in_messages(original_response)
    litellm.error_logs["POST_CALL"] = locals()        # ← 存入已截断版本
    try:
        self.model_call_details["original_response"] = original_response
```

**效果**：`error_logs` 和 `model_call_details` 中均为 KB 级占位符，全局 dict 不再持有大字符串。

---

### Fix 2：`StandardLoggingPayload.response` 截断 base64

**文件**：`litellm/litellm_core_utils/litellm_logging.py`

**问题**：构造 `StandardLoggingPayload` 时，`messages`（请求体）有截断，但 `response`（响应体）没有——导致 173MB 图片数据被转发至每一个 logging callback（Langfuse、DataDog、数据库 span 等）。

```python
# 修复前
response=final_response_obj,

# 修复后
response=truncate_base64_in_messages(final_response_obj),
```

**效果**：与 `messages` 字段保持一致，图片 base64 不再流入 observability 系统。第二次快照中此项从 173MB 降至 22MB，降幅 87%。

---

### Fix 3：`async_set_cache` 延迟序列化

**文件**：`litellm/caching/caching_handler.py`

**问题**：`result.model_dump_json()` 在 `asyncio.create_task()` 调用**之前**求值，创建 479MB JSON 字符串作为协程参数持有，直至 task 执行完毕。事件循环繁忙时多个 pending task 同时持有多个 479MB 字符串。

```python
# 修复前
asyncio.create_task(
    litellm.cache.async_add_cache(
        result.model_dump_json(),   # ← 立即分配 479MB，task 等待期间一直持有
        dynamic_cache_object=self.dual_cache,
        **new_kwargs,
    )
)

# 修复后
async def _cache_result(
    _result=result,
    _dual_cache=self.dual_cache,
    _kwargs=new_kwargs,
):
    await litellm.cache.async_add_cache(
        _result.model_dump_json(),  # ← 在 task 执行时才分配，完成后立即释放
        dynamic_cache_object=_dual_cache,
        **_kwargs,
    )

asyncio.create_task(_cache_result())
```

**效果**：479MB 字符串的生命周期从"task 入队时 → task 执行完"缩短为"task 执行中"，峰值内存显著降低。

---

### Fix 5：超大响应跳过缓存序列化

**文件**：`litellm/caching/caching_handler.py`

**问题（第三轮 memray，16:52）**：高并发下（56 并发图片请求），64 个 `_cache_result` task 同时执行，每个在调用 `async_add_cache`（Redis I/O）期间持有 ~12.5MB JSON 字符串，产生 803MB 峰值。这些超大响应本已被 `InMemoryCache.check_value_size()` 拒绝（默认上限 1MB），序列化纯属浪费。

**关键洞察**：`model_dump_json()` 是同步操作——事件循环同一时刻只有一个 task 处于该同步段。在第一个 `await` 之前 `return`，可确保 GC 在下一个 task 运行前回收字符串，将 N × 12MB 并发峰值降至约 1 × 12MB。

```python
# 修复后
async def _cache_result(_result, _dual_cache, _kwargs):
    json_str = _result.model_dump_json()
    # 超过 per-item 上限（默认 1MB）时，在 await 之前直接返回，
    # 让 GC 在下一个 task 运行前回收该字符串。
    max_bytes = MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB * 1024
    if max_bytes > 0 and len(json_str) > max_bytes:
        return
    await litellm.cache.async_add_cache(
        json_str, dynamic_cache_object=_dual_cache, **_kwargs
    )
```

**效果**：`model_dump_json` 峰值从 803MB（64 并发 × 12.5MB）降至约 12.5MB（同一时刻仅 1 个 task 持有）。

---

## 内存节省对比

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| 单次 Vertex 图片请求峰值 | ~1.07GB | ~100MB 以内 |
| `error_logs` 全局 dict | 持有完整 ~297MB 字符串 | KB 级占位符 |
| logging callbacks 收到的 `response` 字段 | 含完整 base64 图片 | 截断占位符 |
| N 个 pending cache task 持有 | N × 479MB | ~0（延迟到执行时） |
| N 个并发 cache task 执行时持有 | N × 12.5MB = 803MB | ~12.5MB（early return 使 GC 在 task 间回收）|
| 外部系统（Langfuse/DB）存储的 span 体积 | 含完整图片数据 | KB 级元数据 |

---

## 测试覆盖

### `tests/logging_callback_tests/test_unit_test_litellm_logging.py`

| 测试函数 | 验证内容 |
|----------|----------|
| `test_post_call_truncates_base64_in_original_response` | `model_call_details["original_response"]` 中无完整 base64 |
| `test_post_call_truncates_base64_in_error_logs` | `litellm.error_logs["POST_CALL"]` 中无完整 base64（Fix 4 回归测试） |
| `test_post_call_preserves_non_base64_response` | 普通文本响应不被修改 |
| `test_post_call_handles_non_string_original_response` | 非 str/list/dict 类型原样存储，不崩溃 |

### `tests/logging_callback_tests/test_standard_logging_payload.py`

| 测试函数 | 验证内容 |
|----------|----------|
| `test_standard_logging_payload_response_base64_truncated` | `StandardLoggingPayload.response` 中长 base64 被截断 |
| `test_standard_logging_payload_response_short_base64_preserved` | 短 base64（< `MAX_BASE64_LENGTH_FOR_LOGGING`）不被截断 |

### `tests/test_litellm/caching/test_caching_handler.py`

| 测试函数 | 验证内容 |
|----------|----------|
| `test_async_set_cache_defers_model_dump_json` | `create_task` 前 `model_dump_json` 未被调用；task 执行后才调用且早于 `async_add_cache` |
| `test_async_set_cache_skips_oversized_responses` | 超过 per-item 上限的响应不触发 `async_add_cache`（Fix 5 回归测试） |

---

## 相关常量

```python
# litellm/constants.py
MAX_BASE64_LENGTH_FOR_LOGGING = int(os.getenv("MAX_BASE64_LENGTH_FOR_LOGGING", 64))
MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB = int(os.getenv("MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB", 1024))
```

- `MAX_BASE64_LENGTH_FOR_LOGGING`：超过 64 个字符的 base64 payload 会被替换为占位符。设为 `0` 禁用截断。
- `MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB`：缓存条目大小上限（默认 1MB）。Fix 5 以此为阈值决定是否跳过序列化。

---

## Commits

| Commit | 内容 |
|--------|------|
| `75ffd53d4f` | Fix 1（初版）、Fix 2、Fix 3：截断 `model_call_details`、`StandardLoggingPayload.response`、延迟 `model_dump_json` |
| `ea73e3c66a` | 补充测试：7 个测试用例覆盖 Fix 1-3 |
| `7fe68f74e4` | Fix 4：截断提前至 `locals()` 前，修复 `error_logs` 全局 dict 持有大字符串 |
| `903b77a862` | Fix 5：超大响应跳过缓存序列化，803MB → ~12.5MB |

---

## 根本性建议

1. **大文件响应不应进入 logging pipeline**：图片/音频生成类接口的响应体可能达到数百 MB，应在进入 `Logging` 对象之前统一截断，而非在各个下游分别处理。

2. **全局调试 dict 应避免持有业务数据**：`litellm.error_logs` 是模块级 dict，用于保存最近一次调用的现场。这类结构在高并发场景下极易成为内存陷阱。应在存入前统一清理大字段，或改用固定大小的循环缓冲区。

3. **in-memory cache 应设置合理的条目大小上限**：`InMemoryCache` 的 `max_size_per_item` 默认 1MB，可通过环境变量 `MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB` 调整。

4. **`model_call_details` 生命周期应尽早释放**：所有 logging 数据持续保存在 `Logging` 对象中直至所有 callbacks 完成。可考虑在 callbacks 执行完毕后主动 `del` 大字段。
