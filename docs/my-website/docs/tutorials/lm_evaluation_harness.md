import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# LM-Evaluation Harness with TGI

Evaluate LLMs 20x faster with TGI via litellm proxy's `/completions` endpoint. 

This tutorial assumes you're using the `big-refactor` branch of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor)

**Step 1: Start the local proxy**
```shell
$ litellm --model huggingface/bigcode/starcoder
```

Using a custom api base

```shell
$ export HUGGINGFACE_API_KEY=my-api-key #[OPTIONAL]
$ litellm --model huggingface/tinyllama --api_base https://k58ory32yinf1ly0.us-east-1.aws.endpoints.huggingface.cloud
```

OpenAI Compatible Endpoint at http://0.0.0.0:8000

**Step 2: Set OpenAI API Base & Key**
```shell
$ export OPENAI_API_BASE=http://0.0.0.0:8000
```

LM Harness requires you to set an OpenAI API key `OPENAI_API_SECRET_KEY` for running benchmarks
```shell
export OPENAI_API_SECRET_KEY=anything
```

**Step 3: Run LM-Eval-Harness**

```shell
python3 -m lm_eval \
  --model openai-completions \
  --model_args engine=davinci \
  --task crows_pairs_english_age

```

## Debugging 

### Making a test request to your proxy
This command makes a test Completion, ChatCompletion request to your proxy server
```shell
litellm --test
```