# LM-Evaluation Harness with TGI

Evaluate LLMs 20x faster with TGI via litellm proxy's `/completions` endpoint. 

This tutorial assumes you're using the `big-refactor` branch of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/big-refactor)

**Step 1: Start the local proxy**
```shell
$ litellm --model huggingface/bigcode/starcoder
```

OpenAI Compatible Endpoint at http://0.0.0.0:8000

**Step 2: Set OpenAI API Base**
```shell
$ export OPENAI_API_BASE="http://0.0.0.0:8000"
```

**Step 3: Run LM-Eval-Harness**

```shell
python3 -m lm_eval \
  --model openai-completions \
  --model_args engine=davinci \
  --task crows_pairs_english_age

```