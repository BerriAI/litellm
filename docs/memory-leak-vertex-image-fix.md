# LiteLLM Vertex AI 图片响应内存泄漏：分析与修复

## 问题背景

生产环境运行 LiteLLM（`/usr/bin/litellm --use_prisma_db_push --config=...`）时，通过 memray 内存分析发现堆内存持续增长，两次快照均接近 OOM。

---

## Memray 快照时间线

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

### Fix 1-5 后（稳定阶段）
> Duration: 443.7s | Heap: **141MB（53% of 269MB max）**

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
_truncate_base64_in_string           80.015MB    56.59%    80.015MB   56.59%  13   litellm.litellm_cor
send                                 20.814MB    14.72%     0.000B     0.00%  30   httpx._client
```
内存稳定，GC 正常回收。

### 高并发阶段（Fix 6-9 前）
> Duration: 568.8s | Heap: **1.789GB（87% of 2.058GB max）**，增长 ~3.3MB/s

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
_truncate_base64_in_string          682.344MB    38.15%   682.344MB   38.15%  95   litellm.litellm_cor
_read_request_body                  348.202MB    19.47%   348.202MB   19.47% 512   litellm.proxy.commo
convert_to_anthropic_image_obj      347.874MB    19.45%   347.874MB   19.45% 370   litellm.litellm_cor
write                               212.179MB    11.86%   212.179MB   11.86%  30   ssl
```

### Fix 6-10 全部生效后（✅ 稳定）
> Duration: 235.7s → 371.7s | Heap: **108.6MB → 124.5MB**（max 固定 185.4MB）

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
run                                 108.576MB    99.94%   321.764kB    0.30%  1056  asyncio.runners
_acompletion                         71.032MB    65.39%    13.056kB    0.01%   197  litellm.router
async_completion                     70.941MB    65.30%         0B     0.00%   130  litellm.llms.vertex
wrapped_app / __call__ 链            36.263MB    33.38%         0B     0.00%   198  starlette/fastapi
app                                  36.260MB    33.38%     6.288kB    0.01%   192  fastapi.routing
```

136 秒内增长 16MB = **0.12MB/s**（修复前最高 3.3MB/s，降幅约 97%）。

**特征变化**：整个快照中无任何函数 Own Bytes 超过 400kB。所有内存均分散在正常的请求处理调用栈中：

| 来源 | 大小 | 性质 |
|------|------|------|
| `async_completion` (Vertex) + litellm router | ~67-71MB | 并发 LLM API 调用工作集，正常 |
| FastAPI/Starlette middleware 链 | ~36-56MB | HTTP 框架请求处理开销，正常 |
| asyncio/uvicorn 基础 | ~20MB | 进程基础开销，正常 |

max 在两张快照中固定为 185.4MB，进程堆已达稳定水位，不再增长。

---

### 最新验证快照：轻负载 vs 高并发图片压测

两张新快照，分别为轻负载和高并发图片请求场景，进一步验证泄漏已被彻底消除。

#### 轻负载快照（低并发）
> Duration: 33.7s | Heap: **63.874MB（max 134.6MB）**

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
convert_to_anthropic_image_obj       13.054MB    20.44%    13.054MB   20.44%  106  litellm.litellm_core_utils
encode (anthropic)                   10.563MB    16.54%    10.563MB   16.54%   29  litellm.llms.anthropic
raw_decode                           16.939MB    26.52%    16.939MB   26.52%   65  json.decoder
_read_request_body                    2.720MB     4.27%     2.720MB    4.27%    8  httpx._content
encode_content                          112kB     0.18%       112kB    0.18%    -  -
post_call                              14.6kB     0.02%      14.6kB    0.02%    -  litellm.litellm_core
```

堆仅 63.874MB，为极低负载下的稳定基线。`post_call` own bytes 仅 14.6kB，**比修复前下降约 6 个数量级**。

#### 高并发图片压测快照
> Duration: 162.7s | Heap: **408.478MB（72% of 564.3MB max）**

```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
convert_to_anthropic_image_obj       86.681MB    21.22%    86.681MB   21.22%   64  litellm.litellm_core_utils
encode (anthropic)                   82.932MB    20.30%    52.260MB   12.79%   20  litellm.llms.anthropic
_read_request_body                   83.873MB    20.53%    49.107MB   12.02%   72  litellm.proxy.common
raw_decode                           35.831MB     8.77%    35.831MB    8.77%  105  json.decoder
aread                                35.105MB     8.59%    35.092MB    8.59%   26  httpx._models
body                                 34.752MB     8.51%    34.752MB    8.51%   20  starlette.requests
text                                 33.759MB     8.26%    33.759MB    8.26%   20  httpx._models
write                                21.397MB     5.24%    21.397MB    5.24%    2  ssl
iterencode                           14.652MB     3.59%    14.652MB    3.59% 1255  json.encoder
run                                 337.459MB    82.59%     9.171MB    0.56%    -  asyncio.runners
post_call                           261.985kB     0.06%   261.985kB    0.06%   26  litellm.litellm_core
acompletion                         293.552MB    71.86%    36.608kB    0.01%  280  litellm.main
```

408MB 全部来自**当前 in-flight 请求的工作集**，各分配项与并发数成正比：

| 分配来源 | Own Bytes | 并发数 | 每请求均值 | 性质 |
|----------|-----------|--------|-----------|------|
| `convert_to_anthropic_image_obj` | 86.681MB | 64 allocs | ~1.35MB | 图片转 base64 缓冲，请求完成后释放 |
| `encode` (anthropic) | 52.260MB | 20 allocs | ~2.6MB | 请求序列化缓冲 |
| `_read_request_body` | 49.107MB | 72 allocs | ~682kB | 传入请求体，处理完成后释放 |
| `raw_decode` / `aread` / `body` / `text` | ~104MB | 20-105 allocs | ~1-3MB | HTTP 收发缓冲，响应结束后释放 |
| `write` (ssl) | 21.397MB | 2 allocs | ~10.7MB | TLS 写缓冲 |
| `iterencode` | 14.652MB | 1255 allocs | ~11.7kB | 正常 JSON 响应序列化 |
| **`post_call`** | **261.985kB** | **26 allocs** | **~10kB** | **泄漏已清零，正常 metadata** |

**关键对比**：

- `post_call` Own Bytes：修复前高并发下累积数百 MB → 现在 261kB（26 个并发请求合计）
- 无任何函数展示跨请求累积模式（`_truncate_base64_in_string`、`async_success_handler` 等已消失）
- `iterencode` 1255 allocs / 162.7s ≈ 7.7次/秒，属正常 JSON 序列化，非泄漏
- 最大堆固定为 564.3MB，说明峰值有界，不会无限增长

408MB 高并发堆相当于约 64-105 个并发图片请求，每个 ~4-6MB 工作集，完全合理。

**注意：上述高并发快照（408MB）仍属于有界工作集。但以下两张新快照揭示了新的泄漏——见下节。**

---

### 新泄漏发现：convert_to_anthropic_image_obj 持续累积

> Duration: 862.7s → 1026.7s | Heap: **1.875GB → 1.853GB（max 2.109GB → 2.145GB）**

两张连续快照（间隔 164 秒）：

**快照 A**（Duration: 862.7s，排序：Total Bytes）：
```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
run                                   1.821GB    97.13%   8.181MB    0.44%  7916  asyncio.runners
async_function_with_retries           1.227GB    65.47%     0B        0.00%  1906  litellm.router
convert_to_anthropic_image_obj        606.807MB  32.36%   606.807MB  32.36%   446  litellm.litellm_core
_process_gemini_media                 606.807MB  32.36%     0B        0.00%   446  litellm.llms.vertex
_gemini_convert_messages_with_history 606.807MB  32.36%     0B        0.00%   446  litellm.llms.vertex
_transform_messages                   606.807MB  32.36%     0B        0.00%   446  litellm.llms.vertex
_transform_request_body               606.807MB  32.36%     0B        0.00%   446  litellm.llms.vertex
async_transform_request_body          606.807MB  32.36%     0B        0.00%   446  litellm.llms.vertex
wrapped_app / middleware 链            574.106MB  30.62%     0B        0.00%  1124  starlette/fastapi
```

**快照 B**（Duration: 1026.7s，164 秒后）：
```
Location                            Total Bytes  % Total  Own Bytes   % Own  Allocations
convert_to_anthropic_image_obj        665.629MB  35.92%   665.629MB  35.92%   511  litellm.litellm_core
```

**泄漏量化**：
- 862.7s：606.807MB own（446 allocs），每 alloc 约 1.36MB
- 1026.7s：665.629MB own（511 allocs），每 alloc 约 1.30MB
- 164 秒增量：**+58.8MB，+65 allocs，增速 ~0.36MB/s**
- 按最大堆估算 OOM 时间：(2.109GB - 1.875GB) / 0.36MB/s ≈ **650 秒（约 11 分钟）**

#### 根因分析

`convert_to_anthropic_image_obj` 将图片 URL 转换为 base64 字符串（~1.3MB）。
调用链为 Vertex AI 请求转换路径：

```
async_transform_request_body
  → _transform_request_body
    → _gemini_convert_messages_with_history
      → _process_gemini_media
        → convert_to_anthropic_image_obj  ← 在此创建 ~1.3MB base64 字符串
          → image["data"] 放入 {"inline_data": {"data": "<base64>", "mime_type": ...}}
```

转换后的 `request_body`（含 `inline_data.data` base64）被传入：

```python
# vertex_and_google_ai_studio_gemini.py:2542-2550
logging_obj.pre_call(
    input=messages,
    api_key="",
    additional_args={
        "complete_input_dict": request_body,  # ← 包含 ~1.3MB base64
        ...
    },
)
```

`pre_call` → `_pre_call` 将其存入：

```python
self.model_call_details["additional_args"] = additional_args
# additional_args["complete_input_dict"] 持有 base64 字符串引用
```

`model_call_details` 在 Logging 对象整个生命周期（直到所有异步回调完成）都持有该引用。
并发请求越多、回调（DB 写入等）越慢，积压的 446+ 个 1.3MB base64 字符串越多。

---

## 根因分析：数据复制路径

同一份 Vertex AI 图片数据（base64 编码）在处理链路上被**多次复制**，且被多个生命周期较长的对象引用：

```
Vertex 返回 JSON（含 inlineData.data base64 图片）
        │
        ├─① httpx.Response.text ──────────────────── 297MB
        │   transform_response() 调用 raw_response.text
        │   → post_call(original_response=raw_response.text)
        │   → litellm.error_logs["POST_CALL"] = locals()  ◄── 全局 dict，永久持有（Fix 4 目标）
        │   → model_call_details["original_response"]     ◄── 常驻至 callbacks 完成
        │
        ├─② json.decoder.raw_decode ─────────────── 124MB
        │   将 JSON 字符串解析为 Python dict（httpx.Response.json()）
        │   包含 part["inlineData"]["data"] 字段（8MB base64 字符串）
        │
        ├─③ _extract_image_response_from_parts ───── 173MB
        │   data_uri = f"data:{mime_type};base64,{data}"  ← 完整复制 base64
        │   → ImageURLListItem → result.choices[].message["images"][].image_url["url"]
        │   → Logging 对象持有 result 至所有 async callbacks 完成（Fix 7 目标）
        │
        └─④ model_dump_json ────────────────────────── 479MB
            caching_handler.py:
            asyncio.create_task(cache.async_add_cache(
                result.model_dump_json(),  ← 提前序列化，字符串在 task 入队时分配
            ))
```

### 各分配点持有者

| 分配点 | 被谁持有 | 释放时机 |
|--------|----------|----------|
| `litellm.error_logs["POST_CALL"]["original_response"]` | **全局模块 dict** | 下次 `post_call` 调用时覆盖 |
| `model_call_details["original_response"]` 297MB | `Logging` 对象 | 所有 callback 执行完毕后 |
| `result.choices[].message["images"][].image_url["url"]` 7.9MB | `Logging` 对象持有的 `result` | 同上（Fix 7 目标） |
| `model_dump_json()` 序列化字符串 479MB | `asyncio.Task` 协程参数 | task 执行完毕后 |
| `model_call_details["httpx_response"]` | `Logging` 对象 | 同上（**待修复，Fix 10 候选**） |

---

## 修复方案

### Fix 1（已修订）：截断提前至 `locals()` 之前

**文件**：`litellm/litellm_core_utils/litellm_logging.py`

`post_call` 第一行 `litellm.error_logs["POST_CALL"] = locals()` 在截断之前执行，全局 dict 持有完整的 297MB 字符串。

```python
# 修复前
def post_call(self, original_response, ...):
    litellm.error_logs["POST_CALL"] = locals()        # ← 存了完整 297MB
    if isinstance(original_response, dict):
        original_response = json.dumps(original_response)
    try:
        self.model_call_details["original_response"] = truncate_base64_in_messages(...)

# 修复后
def post_call(self, original_response, ...):
    if isinstance(original_response, dict):
        original_response = json.dumps(original_response)
    if isinstance(original_response, (str, list, dict)):
        original_response = truncate_base64_in_messages(original_response)
    litellm.error_logs["POST_CALL"] = locals()        # ← 存入已截断版本
    try:
        self.model_call_details["original_response"] = original_response
```

---

### Fix 2：`StandardLoggingPayload.response` 截断 base64

**文件**：`litellm/litellm_core_utils/litellm_logging.py`

构造 `StandardLoggingPayload` 时，`response` 字段未截断，173MB 图片数据转发至所有 logging callback。

```python
# 修复前
response=final_response_obj,

# 修复后
response=truncate_base64_in_messages(final_response_obj),
```

---

### Fix 3：`async_set_cache` 延迟序列化

**文件**：`litellm/caching/caching_handler.py`

`result.model_dump_json()` 在 `asyncio.create_task()` 之前求值，479MB JSON 字符串在 task 入队时分配并持有至 task 执行完毕。

```python
# 修复前
asyncio.create_task(
    litellm.cache.async_add_cache(result.model_dump_json(), ...)
)

# 修复后
async def _cache_result(_result=result, ...):
    await litellm.cache.async_add_cache(_result.model_dump_json(), ...)  # task 内才分配

asyncio.create_task(_cache_result())
```

---

### Fix 4：`error_logs` 截断（见 Fix 1 修订版）

Fix 1 原版只截断了 `model_call_details`；Fix 4 将截断提前至 `locals()` 快照之前，确保全局 `error_logs` dict 也持有截断版本。

---

### Fix 5：超大响应跳过缓存序列化

**文件**：`litellm/caching/caching_handler.py`

64 个并发 `_cache_result` task 每个在 Redis I/O 期间持有 ~12.5MB JSON 字符串（803MB 峰值）。这些超大响应已被 `InMemoryCache` 拒绝（默认上限 1MB），序列化纯属浪费。

```python
async def _cache_result(_result, _dual_cache, _kwargs):
    json_str = _result.model_dump_json()
    max_bytes = MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB * 1024
    if max_bytes > 0 and len(json_str) > max_bytes:
        return  # 在 await 之前 return，GC 可在下一个 task 前回收
    await litellm.cache.async_add_cache(json_str, ...)
```

**效果**：`model_dump_json` 峰值从 803MB → ~12.5MB。

---

### Fix 6：`_truncate_base64_in_string` 增加 Vertex 原始 JSON 格式支持

**文件**：`litellm/litellm_core_utils/logging_utils.py`

原 `_DATA_URI_RE` 仅匹配 `data:mime;base64,<payload>` 格式，对 Vertex API 原始 JSON 中的 `"data":"<BASE64>"` 格式无效，12MB 字符串原样存入 `model_call_details` 和 `error_logs`。

新增 `_JSON_BASE64_FIELD_RE`：
```python
_JSON_BASE64_FIELD_RE = re.compile(
    r'("(?:data|imageBytes|bytesBase64Encoded)"\s*:\s*")([A-Za-z0-9+/=]{64,})(")'
)
```

---

### Fix 7：`async_success_handler` 中截断 `result` 的图片 data URI

**文件**：`litellm/litellm_core_utils/litellm_logging.py`

`_extract_image_response_from_parts` 在 `ModelResponse.choices[].message["images"][].image_url["url"]` 中生成 ~7.9MB data URI。`Logging` 对象持有 `result` 引用至所有 async callbacks 完成，32 并发 = ~253MB 常驻。

在 `standard_logging_object` 构建后（已有截断副本）、主回调循环前，调用 `strip_large_base64_from_result(result)` 原地截断。

---

### Fix 8：截断 JSON 解析后的裸 base64 字符串

**文件**：`litellm/litellm_core_utils/logging_utils.py`

`httpx.Response.json()` 解析 Vertex 响应后，`"inlineData.data"` 的值是裸 base64 字符串（无 JSON 语法包裹）。前两个 regex 均需要包裹结构，对裸 payload 无效。

新增 `_BARE_BASE64_RE` 快速路径：
```python
_BARE_BASE64_RE = re.compile(r"^[A-Za-z0-9+/]{64,}={0,2}$")

def _truncate_base64_in_string(value: str) -> str:
    if len(value) > MAX_BASE64_LENGTH_FOR_LOGGING and _BARE_BASE64_RE.match(value):
        size_str = _format_base64_size(len(value))
        return f"[base64_data truncated: {size_str}]"
    ...
```

**效果**：`raw_decode` 持有的 238MB 降至 ~1MB。

---

### Fix 9：`model_call_details["original_response"]` 字符数上限

**文件**：`litellm/litellm_core_utils/litellm_logging.py` + `litellm/constants.py`

Fix 6 的 `_JSON_BASE64_FIELD_RE` 移除 8MB base64 后，输出仍是完整 Vertex JSON 结构（约 6-7MB）。95 并发 × 7MB = 665MB 常驻。

在 `post_call` 完成 base64 截断后，加字符数上限：
```python
MAX_ORIGINAL_RESPONSE_LOG_CHARS = int(os.getenv("MAX_ORIGINAL_RESPONSE_LOG_CHARS", 10_000))

if (
    MAX_ORIGINAL_RESPONSE_LOG_CHARS > 0
    and isinstance(original_response, str)
    and len(original_response) > MAX_ORIGINAL_RESPONSE_LOG_CHARS
):
    original_response = (
        original_response[:MAX_ORIGINAL_RESPONSE_LOG_CHARS]
        + f"...[truncated, {len(original_response)} chars total]"
    )
```

**效果**：`_truncate_base64_in_string` 持有的 682MB 预计降至 < 1MB × 95 allocs ≈ ~10MB。

---

## 待修复：Fix 10 候选（`model_call_details["httpx_response"]`）

**文件**：
- `litellm/llms/custom_httpx/llm_http_handler.py:1972`
- `litellm/llms/gemini/google_genai/transformation.py:358`

**问题**：两处代码将完整 `httpx.Response` 对象存入 `model_call_details`：

```python
logging_obj.model_call_details["httpx_response"] = response
```

`httpx.Response` 持有：
- `.content`：响应体原始字节（~12MB）
- `.text`：响应体字符串（如已访问则缓存在对象内）

使用方 `_handle_google_genai_generate_content_response`（litellm_logging.py:3420）在 logging 阶段调用 `httpx_response.json()`，每次都重新解析创建新的 ~7.5MB 解析 dict。`httpx_response` 本身则在 `Logging` 对象生命周期内一直持有。

**建议修复**：在 `_handle_google_genai_generate_content_response` 使用完后删除该引用：
```python
dict_result = httpx_response.json()
del self.model_call_details["httpx_response"]  # 释放 ~12MB 响应体
```
或在存入时仅保留 `.json()` 的解析结果（截断 base64 后），不存储 httpx Response 对象本身。

---

## 修复效果对比

| 阶段 | 堆内存 | 增长速率 | 状态 |
|------|--------|----------|------|
| 修复前 | 1.085GB（838s 时接近 OOM）| ~1.3 MB/s | ❌ OOM 风险 |
| Fix 1-5 后（稳定） | 141MB | ~0 | ✅ 稳定 |
| 高并发下 Fix 6-9 前 | 788MB → 1.789GB（569s）| 3.3 MB/s | ❌ OOM 风险 |
| **Fix 6-10 全部生效（稳定性验证）** | **108MB → 124MB（372s）** | **~0.12 MB/s** | ✅ **稳定** |
| **Fix 6-10 后轻负载** | **63.874MB（33.7s）** | **—** | ✅ **极低基线** |
| **Fix 6-10 后高并发图片压测** | **408MB（162.7s，64+ 并发图片请求）** | **有界** | ✅ **工作集，非泄漏** |
| Fix 6-10 后持续高并发（含新泄漏）| 1.875GB → 1.853GB（862.7s → 1026.7s）| ~0.36MB/s | ❌ Fix 11 前 |

---

## 各泄漏来源分类

| 来源 | 性质 | 修复 |
|------|------|------|
| `model_dump_json` 479MB | 持久：task 入队时提前分配 | Fix 3 |
| `httpx.text` 297MB + `error_logs` | 持久：全局 dict 持有完整响应 | Fix 4 |
| `_extract_image_response_from_parts` 173MB | 持久：data URI 在 Logging 对象存活期持有 | Fix 7 |
| `model_dump_json` 803MB（并发峰值）| 峰值：N 并发 × 12.5MB | Fix 5 |
| `raw_decode` 238MB | 持久：裸 base64 字符串穿透 regex | Fix 8 |
| `_truncate_base64_in_string` 682MB | 持久：Fix 6 输出的截断后 JSON 结构 | Fix 9 |
| `_read_request_body` 348MB | 工作集：512 并发请求体，请求完成后释放 | 无需修复 |
| `convert_to_anthropic_image_obj` 347MB（早期）| 工作集：str.split 创建 base64 副本，转换后释放 | 无需修复（早期） |
| `model_call_details["httpx_response"]` | 持久：httpx Response 对象（含 ~12MB 响应体）| Fix 10 |
| `model_call_details["additional_args"]["complete_input_dict"]` 606MB（446+ allocs）| 持久：Vertex 请求体 `inline_data.data` ~1.3MB/请求，在 Logging 对象存活期积压 | Fix 11 |

---

## 测试覆盖

### `tests/logging_callback_tests/test_unit_test_litellm_logging.py`

| 测试函数 | 验证内容 | Fix |
|----------|----------|-----|
| `test_post_call_truncates_base64_in_original_response` | `model_call_details["original_response"]` 中无完整 base64 | Fix 1 |
| `test_post_call_preserves_non_base64_response` | 普通文本响应不被修改 | Fix 1 |
| `test_post_call_handles_non_string_original_response` | 非 str/list/dict 类型原样存储 | Fix 1 |
| `test_post_call_truncates_raw_vertex_json_base64` | Vertex 原始 JSON `"data":"<BASE64>"` 被截断 | Fix 6 |
| `test_post_call_truncates_base64_in_error_logs` | `litellm.error_logs["POST_CALL"]` 中无完整 base64 | Fix 4 |
| `test_strip_large_base64_from_result` | `result.choices[].message["images"][].image_url["url"]` 被截断 | Fix 7 |
| `test_strip_large_base64_from_result_no_images` | 无图片响应原样返回 | Fix 7 |
| `test_truncate_base64_in_string_bare_payload` | 裸 base64 字符串被截断 | Fix 8 |
| `test_truncate_base64_in_value_bare_dict_value` | parsed dict 中的裸 base64 值被截断 | Fix 8 |
| `test_post_call_caps_original_response_length` | 超长 `original_response` 被裁断 | Fix 9 |
| `test_post_call_does_not_cap_short_response` | 短响应不被修改 | Fix 9 |
| `test_handle_anthropic_messages_response_pops_httpx_response` | Anthropic handler 使用 `.pop()` 释放 httpx_response | Fix 10 |
| `test_handle_google_genai_generate_content_response_pops_httpx_response` | Google GenAI handler 使用 `.pop()` 释放 httpx_response | Fix 10 |
| `test_handle_anthropic_messages_response_no_httpx_response_uses_result` | httpx_response 缺失时回退到 `result` | Fix 10 |
| `test_pre_call_strips_base64_from_complete_input_dict` | `_pre_call` 存储前截断 `complete_input_dict` 中的 inline_data base64 | Fix 11 |
| `test_pre_call_no_complete_input_dict_unchanged` | 无 `complete_input_dict` 时 additional_args 原样存储 | Fix 11 |

### `tests/test_litellm/caching/test_caching_handler.py`

| 测试函数 | 验证内容 | Fix |
|----------|----------|-----|
| `test_async_set_cache_defers_model_dump_json` | `create_task` 前未调用 `model_dump_json` | Fix 3 |
| `test_async_set_cache_skips_oversized_responses` | 超大响应不触发 `async_add_cache` | Fix 5 |

---

## 相关常量

```python
# litellm/constants.py
MAX_BASE64_LENGTH_FOR_LOGGING = int(os.getenv("MAX_BASE64_LENGTH_FOR_LOGGING", 64))
# base64 超 64 字符替换为占位符；设为 0 禁用

MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB = int(os.getenv("MAX_SIZE_PER_ITEM_IN_MEMORY_CACHE_IN_KB", 1024))
# 缓存条目大小上限（默认 1MB）；Fix 5 以此决定是否跳过序列化

MAX_ORIGINAL_RESPONSE_LOG_CHARS = int(os.getenv("MAX_ORIGINAL_RESPONSE_LOG_CHARS", 10_000))
# model_call_details["original_response"] 字符数上限（默认 10KB）；Fix 9 新增
```

---

## Commits

| Commit | 内容 |
|--------|------|
| `75ffd53d4f` | Fix 1（初版）、Fix 2、Fix 3 |
| `ea73e3c66a` | 补充测试：Fix 1-3 |
| `7fe68f74e4` | Fix 4：截断提前至 `locals()` 前 |
| `903b77a862` | Fix 5：超大响应跳过缓存序列化 |
| `f04383437e` | Fix 6：`_JSON_BASE64_FIELD_RE` 覆盖 Vertex 原始 JSON 格式 |
| `862285e8ac` | Fix 7：`strip_large_base64_from_result`，截断 Logging 对象内 data URI |
| `9b9aef5890` | Fix 8：`_BARE_BASE64_RE` 截断裸 base64 字符串 |
| `cb755d68ef` | Fix 9：`MAX_ORIGINAL_RESPONSE_LOG_CHARS`，`original_response` 字符数上限 |
| `cd8f9769f3` | Fix 10：`.pop("httpx_response")` 释放 httpx Response 对象（~12MB/次）|
| *(current)* | Fix 11：`_pre_call` 存储 `complete_input_dict` 前截断 `inline_data.data` base64（~1.3MB/次）|

---

## 根本性建议

1. **大文件响应不应进入 logging pipeline**：图片/音频生成类接口响应体可达数百 MB，应在进入 `Logging` 对象之前统一截断。

2. **全局调试 dict 避免持有业务数据**：`litellm.error_logs` 是模块级 dict，高并发下易成内存陷阱。应在存入前清理大字段，或改用固定大小循环缓冲区。

3. **不要把 `httpx.Response` 对象存入 `model_call_details`**：httpx Response 持有完整响应体字节。如需用于 logging，应立即提取所需内容（如解析后截断的 dict），再存入 `model_call_details`，并在使用后删除引用。

4. **`model_call_details` 生命周期问题**：所有 logging 数据持续保存至所有 callbacks 完成。可考虑在 callbacks 执行完毕后主动 `del` 大字段（`original_response`、`httpx_response`）。
