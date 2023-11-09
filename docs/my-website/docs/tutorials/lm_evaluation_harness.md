import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Benchmark LLMs - LM Harness, Flask

## LM Harness Benchmarks
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

## FLASK - Fine-grained Language Model Evaluation 
Use litellm to evaluate any LLM on FLASK https://github.com/kaistAI/FLASK 

**Step 1: Start the local proxy**
```shell
$ litellm --model huggingface/bigcode/starcoder
```

**Step 2: Set OpenAI API Base & Key**
```shell
$ export OPENAI_API_BASE=http://0.0.0.0:8000
```

**Step 3 Run with FLASK** 

```shell
git clone https://github.com/kaistAI/FLASK
```
```shell
cd FLASK/gpt_review
```

Run the eval 
```shell
python gpt4_eval.py -q '../evaluation_set/flask_evaluation.jsonl'
```

## Debugging 

### Making a test request to your proxy
This command makes a test Completion, ChatCompletion request to your proxy server
```shell
litellm --test
```