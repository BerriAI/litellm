# Forwarding assistant reasoning to hosted vLLM

Some vLLM backends and chat templates accept previous assistant reasoning alongside content and tool calls. Set `forward_reasoning_content: true` on an individual model entry to forward the `reasoning_content` field supplied by the client

```yaml
model_list:
  - model_name: reasoning-history
    litellm_params:
      model: hosted_vllm/your-served-model
      api_base: http://localhost:8000/v1
      forward_reasoning_content: true
  - model_name: default-history
    litellm_params:
      model: hosted_vllm/your-served-model
      api_base: http://localhost:8000/v1
      forward_reasoning_content: false
```

The default is false. Omitting the option or setting it to false retains the existing removal of assistant `reasoning_content`. The option is local to each request, applies only to `hosted_vllm`, and is not sent to the backend. It can also be passed to `litellm.completion` or `litellm.acompletion`

When enabled, LiteLLM forwards the supplied field without trimming it or inserting it into visible `content`. Content and tool calls keep their existing transformations. It does not reconstruct missing reasoning or convert Anthropic `thinking_blocks`, signatures, or redacted thinking. Their existing handling is unchanged

For Responses requests routed through Chat Completions, also set `use_chat_completions_api: true`. The option preserves the `reasoning_content` produced by that bridge from supported Responses reasoning input. It does not change native vLLM Responses requests or make opaque encrypted reasoning portable

## Backend compatibility

Enable this only after checking both your vLLM request parser and model chat template. A backend returning reasoning in its responses does not necessarily accept reasoning in previous assistant messages

The vLLM v0.12.0 and v0.13.0 chat parsers accept `reasoning` and `reasoning_content` and expose both names to the template. Newer vLLM versions may normalize the deprecated input name `reasoning_content` to `reasoning` before rendering. LiteLLM forwards its existing `reasoning_content` field, without adding a duplicate `reasoning` field or selecting behavior by model name. Older releases, vendor forks and custom templates need separate verification

Sources: [vLLM v0.12.0 chat parser](https://github.com/vllm-project/vllm/blob/v0.12.0/vllm/entrypoints/chat_utils.py#L1532), [vLLM v0.13.0 chat parser](https://github.com/vllm-project/vllm/blob/v0.13.0/vllm/entrypoints/chat_utils.py)

## Template policy and performance

`forward_reasoning_content` controls transport. A model-specific setting such as `chat_template_kwargs.preserve_thinking` controls which received history its template uses. In the Qwen3.8-Flash-Next template, `preserve_thinking: false` removes reasoning from earlier user turns but still retains reasoning within the current tool sequence. It does not mean that every assistant reasoning field should be deleted

See [Qwen3.8-Flash-Next preserved thinking](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#disable-preserved-thinking)

Forwarding history can change the rendered prompt and token prefix. Boundary whitespace normalization may have no effect when the template trims that field. Validate rendered tokens with the actual tokenizer and template before drawing cache conclusions. This option provides no measured latency, cache-hit or reasoning-quality guarantee
