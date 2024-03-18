# LiteLLM Docker Proxy

LiteLLM Proxyは、100以上のLLMを統一されたインターフェースで呼び出し、仮想キーやユーザーごとに支出を追跡し、予算を設定できるプロキシサーバーです。

<div align="center">

 | [日本語](README.Proxy.JP.md) | [English](README.Proxy.md) |

</div>

## 主な機能

- **統一されたインターフェース**: Huggingface、Bedrock、TogetherAIなど、100以上のLLMをOpenAIの`ChatCompletions`および`Completions`形式で呼び出すことができます。
- **コスト追跡**: 仮想キーによる認証、支出の追跡、予算の設定が可能です。
- **負荷分散**: 同じモデルの複数のモデルとデプロイメント間で負荷分散を行います。LiteLLM proxyは、負荷テスト中に1.5k以上のリクエスト/秒を処理できます。

## クイックスタート

CLIを使用してLiteLLM Proxyを素早く起動できます。

```bash
$ pip install 'litellm[proxy]'
```

Dockerを使用する場合は、以下のコマンドでLiteLLMのコンテナを起動します。

```bash
docker-compose -f docker\docker-compose.gemi.yml up --build
```

コンテナが正常に起動すると、以下のようなログが表示されます。

```bash
litellm-1  | INFO:     Started server process [1]
litellm-1  | INFO:     Waiting for application startup.
litellm-1  |
litellm-1  | #------------------------------------------------------------#
litellm-1  | #                                                            #
litellm-1  | #              'I don't like how this works...'               #
litellm-1  | #        https://github.com/BerriAI/litellm/issues/new        #
litellm-1  | #                                                            #
litellm-1  | #------------------------------------------------------------#
litellm-1  |
litellm-1  |  Thank you for using LiteLLM! - Krrish & Ishaan
litellm-1  |
litellm-1  |
litellm-1  |
litellm-1  | Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
litellm-1  |
litellm-1  |
litellm-1  | INFO:     Application startup complete.
litellm-1  | INFO:     Uvicorn running on http://0.0.0.0:4000 (Press CTRL+C to quit)
```

デモスクリプトを実行するには、以下のコマンドを使用します。

```bash
python docker\demo\demo_openai.py
```

デモスクリプトの実行結果が以下のように表示されます。

```bash
ChatCompletion(id='chatcmpl-7ef51102-505c-4c54-9e5a-783f6d4d0401', choices=[Choice(finish_reason='stop', index=1, logprobs=None, message=ChatCompletionMessage(content="In realms of words, a dance takes place,\nA symphony of rhythm, grace.\nEach syllable a note so fine,\nWeaving stories, making hearts entwine.\n\nFrom whispers soft to thunder's roar,\nWords paint worlds, forevermore.\nThey evoke emotions, deep and true,\nGuiding us through life's every hue.", role='assistant', function_call=None, tool_calls=None))], created=1710775833, model='gemini/gemini-pro', object='chat.completion', system_fingerprint=None, usage=CompletionUsage(completion_tokens=67, prompt_tokens=10, total_tokens=77))
-------------
In realms of words, a dance takes place,
A symphony of rhythm, grace.
Each syllable a note so fine,
Weaving stories, making hearts entwine.

From whispers soft to thunder's roar,
Words paint worlds, forevermore.
They evoke emotions, deep and true,
Guiding us through life's every hue.
```
