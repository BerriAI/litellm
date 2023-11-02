# LM-Evaluation Harness with TGI

Evaluate LLMs 20x faster with TGI via litellm proxy's `/completions` endpoint. 

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
$ python3 main.py \
  --model gpt3 \
  --model_args engine=huggingface/bigcode/starcoder \
  --tasks hellaswag
```