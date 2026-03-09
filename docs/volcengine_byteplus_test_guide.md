# Volcengine / BytePlus Provider 自测指引

本文档提供 `volcengine`、`volcengine_plan`、`byteplus`、`byteplus_plan` 四个 provider 的逐步自测流程。

---

## 前置条件

```bash
# 1. 切到功能分支
git checkout feat/add-volcengine-plan-byteplus-providers

# 2. 安装依赖（二选一）
make install-dev
# 或
pip install -e .
pip install pytest openai httpx
```

---

## Step 1: Import 安全检查

验证新增代码不会导致 `from litellm import *` 报错。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "from litellm import *; print('OK')"
```

**预期输出：** `OK`

---

## Step 2: 循环依赖检查

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python tests/documentation_tests/test_circular_imports.py
```

**预期输出：** 正常输出 type hints 列表，无 `ImportError` 或 `CircularImportError`。

---

## Step 3: LlmProviders 枚举验证

确认 3 个新枚举值已注册。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
from litellm.types.utils import LlmProviders
assert LlmProviders.VOLCENGINE_PLAN.value == 'volcengine_plan'
assert LlmProviders.BYTEPLUS.value == 'byteplus'
assert LlmProviders.BYTEPLUS_PLAN.value == 'byteplus_plan'
print('LlmProviders enum: OK')
"
```

---

## Step 4: 模型集注册验证

确认所有模型从 `model_prices_and_context_window.json` 正确加载。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
import litellm

# volcengine 合并的 2 个新模型
assert 'doubao-seed-2-0-pro-260215' in litellm.volcengine_models
assert 'kimi-k2-5-260127' in litellm.volcengine_models
print(f'volcengine_models ({len(litellm.volcengine_models)}): OK')

# volcengine_plan: 9 个模型
assert len(litellm.volcengine_plan_models) == 9
assert 'volcengine_plan/ark-code-latest' in litellm.volcengine_plan_models
assert 'volcengine_plan/doubao-seed-code' in litellm.volcengine_plan_models
assert 'volcengine_plan/Doubao-Seed-2.0-Code' in litellm.volcengine_plan_models
print(f'volcengine_plan_models ({len(litellm.volcengine_plan_models)}): OK')

# byteplus: 3 个模型
assert len(litellm.byteplus_models) == 3
assert 'byteplus/seed-2-0-mini-260215' in litellm.byteplus_models
print(f'byteplus_models ({len(litellm.byteplus_models)}): OK')

# byteplus_plan: 4 个模型
assert len(litellm.byteplus_plan_models) == 4
assert 'byteplus_plan/ark-code-latest' in litellm.byteplus_plan_models
print(f'byteplus_plan_models ({len(litellm.byteplus_plan_models)}): OK')
"
```

---

## Step 5: Provider 解析验证

确认 `model` 前缀能正确解析到对应 provider 和 base URL。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

tests = [
    ('volcengine/doubao-seed-2-0-pro-260215',  'volcengine',      'https://ark.cn-beijing.volces.com/api/v3'),
    ('volcengine_plan/ark-code-latest',         'volcengine_plan', 'https://ark.cn-beijing.volces.com/api/coding/v3'),
    ('byteplus/glm-4-7-251222',                 'byteplus',        'https://ark.ap-southeast.bytepluses.com/api/v3'),
    ('byteplus_plan/kimi-k2.5',                 'byteplus_plan',   'https://ark.ap-southeast.bytepluses.com/api/coding/v3'),
]

for model_str, expected_provider, expected_base in tests:
    model, provider, key, api_base = get_llm_provider(model=model_str, api_base=None, api_key=None)
    assert provider == expected_provider, f'{model_str}: provider={provider}, expected={expected_provider}'
    assert api_base == expected_base, f'{model_str}: api_base={api_base}, expected={expected_base}'
    print(f'{model_str}: provider={provider}, api_base={api_base}  OK')
"
```

---

## Step 6: API Key 共享验证

确认 volcengine/volcengine_plan 共享 key，byteplus/byteplus_plan 共享 key。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
import os
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

# volcengine + volcengine_plan 共享 VOLCENGINE_API_KEY
os.environ['VOLCENGINE_API_KEY'] = 'volc-test-key-123'
_, _, key1, _ = get_llm_provider(model='volcengine/doubao-seed-2-0-pro-260215', api_base=None, api_key=None)
_, _, key2, _ = get_llm_provider(model='volcengine_plan/ark-code-latest', api_base=None, api_key=None)
assert key1 == 'volc-test-key-123', f'volcengine key: {key1}'
assert key2 == 'volc-test-key-123', f'volcengine_plan key: {key2}'
print(f'volcengine + volcengine_plan 共享 VOLCENGINE_API_KEY: OK')

# 也支持 ARK_API_KEY
del os.environ['VOLCENGINE_API_KEY']
os.environ['ARK_API_KEY'] = 'ark-test-key-456'
_, _, key3, _ = get_llm_provider(model='volcengine_plan/ark-code-latest', api_base=None, api_key=None)
assert key3 == 'ark-test-key-456', f'volcengine_plan ARK key: {key3}'
print(f'volcengine_plan 支持 ARK_API_KEY: OK')
del os.environ['ARK_API_KEY']

# byteplus + byteplus_plan 共享 BYTEPLUS_API_KEY
os.environ['BYTEPLUS_API_KEY'] = 'bp-test-key-789'
_, _, key4, _ = get_llm_provider(model='byteplus/glm-4-7-251222', api_base=None, api_key=None)
_, _, key5, _ = get_llm_provider(model='byteplus_plan/kimi-k2.5', api_base=None, api_key=None)
assert key4 == 'bp-test-key-789', f'byteplus key: {key4}'
assert key5 == 'bp-test-key-789', f'byteplus_plan key: {key5}'
print(f'byteplus + byteplus_plan 共享 BYTEPLUS_API_KEY: OK')
del os.environ['BYTEPLUS_API_KEY']
"
```

---

## Step 7: Thinking 参数映射验证

确认 4 个 provider 都正确继承了 VolcEngine 的 thinking 参数处理。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
from litellm.utils import get_optional_params

providers = [
    ('doubao-seed-2-0-pro-260215', 'volcengine'),
    ('ark-code-latest',            'volcengine_plan'),
    ('glm-4-7-251222',             'byteplus'),
    ('kimi-k2.5',                  'byteplus_plan'),
]

for model, provider in providers:
    # thinking enabled -> extra_body
    params = get_optional_params(
        model=model, custom_llm_provider=provider,
        thinking={'type': 'enabled'}, drop_params=False,
    )
    assert params.get('extra_body', {}).get('thinking') == {'type': 'enabled'}, \
        f'{provider}: thinking enabled failed: {params}'

    # thinking disabled -> extra_body
    params2 = get_optional_params(
        model=model, custom_llm_provider=provider,
        thinking={'type': 'disabled'}, drop_params=False,
    )
    assert params2.get('extra_body', {}).get('thinking') == {'type': 'disabled'}, \
        f'{provider}: thinking disabled failed: {params2}'

    # thinking None -> not in extra_body
    params3 = get_optional_params(
        model=model, custom_llm_provider=provider,
        thinking=None, drop_params=False,
    )
    assert 'extra_body' not in params3 or 'thinking' not in params3.get('extra_body', {}), \
        f'{provider}: thinking None should be ignored: {params3}'

    print(f'{provider}/{model}: thinking mapping OK')
"
```

---

## Step 8: Mock E2E Completion 验证

模拟完整调用链，确认 model 名称正确传递给 OpenAI SDK。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
from unittest.mock import MagicMock, patch
from openai import OpenAI
from litellm import completion
from litellm.types.utils import ModelResponse

models = [
    ('volcengine/doubao-seed-2-0-pro-260215', 'doubao-seed-2-0-pro-260215'),
    ('volcengine_plan/ark-code-latest',       'ark-code-latest'),
    ('byteplus/glm-4-7-251222',               'glm-4-7-251222'),
    ('byteplus_plan/kimi-k2.5',               'kimi-k2.5'),
]

for full_model, expected_model in models:
    client = OpenAI(api_key='test_key')
    mock_raw = MagicMock()
    mock_raw.headers = {
        'x-request-id': '123',
        'openai-organization': 'org-123',
        'x-ratelimit-limit-requests': '100',
        'x-ratelimit-remaining-requests': '99',
    }
    mock_raw.parse.return_value = ModelResponse()

    with patch.object(client.chat.completions.with_raw_response, 'create', mock_raw) as mock_create:
        completion(
            model=full_model,
            messages=[{'role': 'user', 'content': 'hello'}],
            client=client,
        )
        mock_create.assert_called_once()
        actual_model = mock_create.call_args.kwargs['model']
        assert actual_model == expected_model, f'{full_model}: got model={actual_model}'
        print(f'{full_model} -> SDK model={actual_model}: OK')
"
```

---

## Step 9: Mock E2E Completion with Thinking 参数验证

确认 thinking 参数通过 extra_body 正确传递到 OpenAI SDK。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -c "
from unittest.mock import MagicMock, patch
from openai import OpenAI
from litellm import completion
from litellm.types.utils import ModelResponse

models = [
    'volcengine_plan/Doubao-Seed-2.0-Code',
    'byteplus_plan/doubao-seed-code',
]

for full_model in models:
    client = OpenAI(api_key='test_key')
    mock_raw = MagicMock()
    mock_raw.headers = {
        'x-request-id': '123',
        'openai-organization': 'org-123',
        'x-ratelimit-limit-requests': '100',
        'x-ratelimit-remaining-requests': '99',
    }
    mock_raw.parse.return_value = ModelResponse()

    with patch.object(client.chat.completions.with_raw_response, 'create', mock_raw) as mock_create:
        completion(
            model=full_model,
            messages=[{'role': 'user', 'content': 'hello'}],
            thinking={'type': 'enabled'},
            client=client,
        )
        mock_create.assert_called_once()
        extra_body = mock_create.call_args.kwargs.get('extra_body', {})
        assert extra_body.get('thinking') == {'type': 'enabled'}, \
            f'{full_model}: extra_body={extra_body}'
        print(f'{full_model}: thinking in extra_body OK')
"
```

---

## Step 10: 单元测试（全量）

运行所有新增和已有的 volcengine/byteplus 测试。

```bash
LITELLM_LOCAL_MODEL_COST_MAP=true python -m pytest \
    tests/test_litellm/llms/volcengine/ \
    tests/test_litellm/llms/volcengine_plan/ \
    tests/test_litellm/llms/byteplus/ \
    tests/test_litellm/llms/byteplus_plan/ \
    -v
```

**预期输出：** `26 passed`

---

## Step 11: 真实 API 调用验证（可选，需要 API Key）

如果有真实 API Key，可以验证端到端调用。

### 11.1 Volcengine Chat

```bash
export VOLCENGINE_API_KEY="your-key-here"

python -c "
from litellm import completion

response = completion(
    model='volcengine/doubao-seed-2-0-pro-260215',
    messages=[{'role': 'user', 'content': '你好，请介绍一下你自己'}],
    max_tokens=100,
)
print(response.choices[0].message.content)
"
```

### 11.2 Volcengine Plan Chat

```bash
export VOLCENGINE_API_KEY="your-key-here"

python -c "
from litellm import completion

response = completion(
    model='volcengine_plan/ark-code-latest',
    messages=[{'role': 'user', 'content': '用 Python 写一个快速排序'}],
    max_tokens=200,
)
print(response.choices[0].message.content)
"
```

### 11.3 BytePlus Chat

```bash
export BYTEPLUS_API_KEY="your-key-here"

python -c "
from litellm import completion

response = completion(
    model='byteplus/glm-4-7-251222',
    messages=[{'role': 'user', 'content': 'Hello, tell me about yourself'}],
    max_tokens=100,
)
print(response.choices[0].message.content)
"
```

### 11.4 BytePlus Plan Chat

```bash
export BYTEPLUS_API_KEY="your-key-here"

python -c "
from litellm import completion

response = completion(
    model='byteplus_plan/kimi-k2.5',
    messages=[{'role': 'user', 'content': 'Write a Python fibonacci function'}],
    max_tokens=200,
)
print(response.choices[0].message.content)
"
```

### 11.5 Thinking 参数真实验证

```bash
export VOLCENGINE_API_KEY="your-key-here"

python -c "
from litellm import completion

response = completion(
    model='volcengine_plan/Doubao-Seed-2.0-Code',
    messages=[{'role': 'user', 'content': '解释 Python GIL 的工作原理'}],
    thinking={'type': 'enabled'},
    max_tokens=500,
)
print(response.choices[0].message.content)
"
```

---

## Step 12: LiteLLM Proxy 验证（可选）

### 12.1 配置 config.yaml

```yaml
model_list:
  - model_name: volcengine-chat
    litellm_params:
      model: volcengine/doubao-seed-2-0-pro-260215
      api_key: os.environ/VOLCENGINE_API_KEY

  - model_name: volcengine-plan-code
    litellm_params:
      model: volcengine_plan/ark-code-latest
      api_key: os.environ/VOLCENGINE_API_KEY

  - model_name: byteplus-chat
    litellm_params:
      model: byteplus/glm-4-7-251222
      api_key: os.environ/BYTEPLUS_API_KEY

  - model_name: byteplus-plan-code
    litellm_params:
      model: byteplus_plan/kimi-k2.5
      api_key: os.environ/BYTEPLUS_API_KEY
```

### 12.2 启动 Proxy

```bash
export VOLCENGINE_API_KEY="your-key-here"
export BYTEPLUS_API_KEY="your-key-here"

litellm --config config.yaml
```

### 12.3 发送请求

```bash
# volcengine_plan
curl http://localhost:4000/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "volcengine-plan-code",
        "messages": [{"role": "user", "content": "Hello"}]
    }'

# byteplus
curl http://localhost:4000/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "byteplus-chat",
        "messages": [{"role": "user", "content": "Hello"}]
    }'
```

---

## 自测检查清单

| # | 检查项 | 命令/步骤 | 预期结果 |
|---|--------|-----------|----------|
| 1 | Import 安全 | Step 1 | `OK` |
| 2 | 循环依赖 | Step 2 | 无 ImportError |
| 3 | 枚举注册 | Step 3 | 3 个新枚举值 |
| 4 | 模型集加载 | Step 4 | volcengine +2, plan 9, bp 3, bp_plan 4 |
| 5 | Provider 解析 | Step 5 | 4 个 provider + base URL 正确 |
| 6 | API Key 共享 | Step 6 | volc/plan 共享, bp/bp_plan 共享 |
| 7 | Thinking 映射 | Step 7 | enabled/disabled/None 三种情况 |
| 8 | Mock E2E | Step 8 | 4 个 provider completion 成功 |
| 9 | Mock Thinking E2E | Step 9 | extra_body 正确传递 |
| 10 | 单元测试 | Step 10 | 26 passed |
| 11 | 真实 API（可选） | Step 11 | 返回正常响应 |
| 12 | Proxy（可选） | Step 12 | curl 返回正常响应 |
