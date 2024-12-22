
import Image from '@theme/IdealImage';

# v1.55.8-stable

A new LiteLLM Stable release [just went out](https://github.com/BerriAI/litellm/releases/tag/v1.55.8-stable). Here are 5 updates since v1.52.2-stable. 

`langfuse`, `fallbacks`, `new models`, `azure_storage`

<Image img={require('../../img/langfuse_prmpt_mgmt.png')} />

## Langfuse Prompt Management

This makes it easy to run experiments or change the specific models `gpt-4o` to `gpt-4o-mini` on Langfuse, instead of making changes in your applications. [Start here](https://docs.litellm.ai/docs/proxy/prompt_management)

## Control fallback prompts client-side 

> Claude prompts are different than OpenAI

Pass in prompts specific to model when doing fallbacks. [Start here](https://docs.litellm.ai/docs/proxy/reliability#control-fallback-prompts)


## New Providers / Models

- [NVIDIA Triton](https://developer.nvidia.com/triton-inference-server) `/infer` endpoint. [Start here](https://docs.litellm.ai/docs/providers/triton-inference-server)
- [Infinity](https://github.com/michaelfeil/infinity) Rerank Models [Start here](https://docs.litellm.ai/docs/providers/infinity)


## ✨ Azure Data Lake Storage Support

Send LLM usage (spend, tokens) data to [Azure Data Lake](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-introduction). This makes it easy to consume usage data on other services (eg. Databricks)
 [Start here](https://docs.litellm.ai/docs/proxy/logging#azure-blob-storage)

## Docker Run LiteLLM

```shell
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:litellm_stable_release_branch-v1.55.8-stable
```

## Get Daily Updates

LiteLLM ships new releases every day. [Follow us on LinkedIn](https://www.linkedin.com/company/berri-ai/) to get daily updates. 

