import Image from '@theme/IdealImage';

# Custom Pricing - Sagemaker, etc. 

Use this to register custom pricing (cost per token or cost per second) for models. 

## Quick Start 

Register custom pricing for sagemaker completion + embedding models. 

For cost per second pricing, you **just** need to register `input_cost_per_second`. 

**Step 1: Add pricing to config.yaml**
```yaml
model_list:
  - model_name: sagemaker-completion-model
    litellm_params:
      model: sagemaker/berri-benchmarking-Llama-2-70b-chat-hf-4
      input_cost_per_second: 0.000420
  - model_name: sagemaker-embedding-model
    litellm_params:
      model: sagemaker/berri-benchmarking-gpt-j-6b-fp16
      input_cost_per_second: 0.000420
```

**Step 2: Start proxy**

```bash
litellm /path/to/config.yaml
```

**Step 3: View Spend Logs**

<Image img={require('../../img/spend_logs_table.png')} />