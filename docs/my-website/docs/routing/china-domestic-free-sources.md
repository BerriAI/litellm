# China Domestic Free Sources

Pre-configured free LLM sources for Chinese developers. These sources are accessible from mainland China without VPN.

## Overview

Chinese developers face unique challenges:
- Need VPN to access international LLM APIs
- Prefer local models for latency and privacy

This configuration provides pre-configured free sources that work in China.

## Sources

### SiliconFlow (硅基流动)

- **Website**: https://cloud.siliconflow.cn
- **Free Models**: DeepSeek-R1, Qwen2.5-72B
- **Registration**: Free credits on signup

### Zhipu (智谱)

- **Website**: https://open.bigmodel.cn
- **Free Models**: GLM-4-Flash (永久免费)
- **Registration**: 20M tokens free

## Configuration

```yaml
model_list:
  # SiliconFlow
  - model_name: sf-deepseek-r1
    litellm_params:
      model: siliconflow/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
      api_key: os.environ/SILICONFLOW_API_KEY
      api_base: https://api.siliconflow.cn/v1

  - model_name: sf-qwen2.5-72b
    litellm_params:
      model: siliconflow/Qwen/Qwen2.5-72B-Instruct
      api_key: os.environ/SILICONFLOW_API_KEY
      api_base: https://api.siliconflow.cn/v1

  # Zhipu
  - model_name: glm-4-flash
    litellm_params:
      model: zhipu/glm-4-flash
      api_key: os.environ/ZHIPU_API_KEY
      api_base: https://open.bigmodel.cn/api/paas/v4

  - model_name: glm-4.7-flash
    litellm_params:
      model: zhipu/glm-4.7-flash
      api_key: os.environ/ZHIPU_API_KEY
      api_base: https://open.bigmodel.cn/api/paas/v4
```

## Environment Variables

```bash
export SILICONFLOW_API_KEY="your-key"
export ZHIPU_API_KEY="your-key"
```

## Related

- [Task-Aware Routing](./task-aware-routing.md)
- [Local-First Routing](./local-first-routing.md)
