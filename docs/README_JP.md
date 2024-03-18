<h1 align="center">
        🚅 LiteLLM
    </h1>
    <p align="center">
        <p align="center">OpenAIのフォーマットを使用して、すべてのLLM APIを呼び出す [Bedrock、Huggingface、VertexAI、TogetherAI、Azure、OpenAIなど]
        <br>
    </p>




<h4 align="center"><a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">OpenAIプロキシサーバー（OpenAI Proxy Server）</a> | <a href="https://docs.litellm.ai/docs/enterprise"target="_blank">エンタープライズ層（Enterprise Tier）</a></h4>

LiteLLMが管理するもの:
- プロバイダーの`completion`、`embedding`、`image_generation`エンドポイントへの入力の変換
- [一貫した出力](https://docs.litellm.ai/docs/completion/output)。テキストレスポンスは常に`['choices'][0]['message']['content']`で利用可能
- 複数のデプロイ（Azure/OpenAIなど）にわたるリトライ/フォールバックロジック - [ルーター（Router）](https://docs.litellm.ai/docs/routing)
- プロジェクト、APIキー、モデルごとの予算と料金制限の設定 [OpenAIプロキシサーバー（OpenAI Proxy Server）](https://docs.litellm.ai/docs/simple_proxy)

**安定リリース**: v`1.30.2` 👈 プロキシの推奨安定バージョン。

[**OpenAIプロキシドキュメントにジャンプ（Jump to OpenAI Proxy Docs）**](https://github.com/BerriAI/litellm?tab=readme-ov-file#openai-proxy---docs) <br>
[**サポートされているLLMプロバイダーにジャンプ（Jump to Supported LLM Providers）**](https://github.com/BerriAI/litellm?tab=readme-ov-file#supported-provider-docs)

より多くのプロバイダーをサポート。プロバイダーやLLMプラットフォームが不足している場合は、[機能リクエスト（feature request）](https://github.com/BerriAI/litellm/issues/new?assignees=&labels=enhancement&projects=&template=feature_request.yml&title=%5BFeature%5D%3A+)を上げてください。

# 使用方法（Usage） ([**ドキュメント（Docs）**](https://docs.litellm.ai/docs/))
> [!重要（IMPORTANT）]
> LiteLLM v1.0.0では、`openai>=1.0.0`が必要になりました。移行ガイド（migration guide）は[こちら](https://docs.litellm.ai/docs/migration)


<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colabで開く（Open In Colab）"/>
</a>


```shell
pip install litellm
```

```python
from litellm import completion
import os

## 環境変数の設定 
os.environ["OPENAI_API_KEY"] = "your-openai-key" 
os.environ["COHERE_API_KEY"] = "your-cohere-key" 

messages = [{ "content": "こんにちは、元気ですか？","role": "user"}]

# OpenAIの呼び出し
response = completion(model="gpt-3.5-turbo", messages=messages)

# Cohereの呼び出し
response = completion(model="command-nightly", messages=messages)
print(response)
```

`model=<provider_name>/<model_name>`を使用して、プロバイダーがサポートするモデルを呼び出します。ここにはプロバイダー固有の詳細が含まれる可能性があるため、詳細については[プロバイダーのドキュメント（provider docs）](https://docs.litellm.ai/docs/providers)を参照してください。

## 非同期（Async） ([ドキュメント（Docs）](https://docs.litellm.ai/docs/completion/stream#async-completion))

```python
from litellm import acompletion
import asyncio

async def test_get_response():
    user_message = "こんにちは、元気ですか？"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(model="gpt-3.5-turbo", messages=messages)
    return response

response = asyncio.run(test_get_response())
print(response)
```

## ストリーミング（Streaming） ([ドキュメント（Docs）](https://docs.litellm.ai/docs/completion/stream))
liteLLMはモデルレスポンスのストリーミングをサポートしています。`stream=True`を渡すと、レスポンスにストリーミングイテレーターが返されます。  
ストリーミングはすべてのモデル（Bedrock、Huggingface、TogetherAI、Azure、OpenAIなど）でサポートされています。
```python
from litellm import completion
response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")

# claude 2
response = completion('claude-2', messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")
```

## ロギング監視（Logging Observability） ([ドキュメント（Docs）](https://docs.litellm.ai/docs/observability/callbacks))
LiteLLMは、Langfuse、DynamoDB、S3バケット、LLMonitor、Helicone、Promptlayer、Traceloop、Athina、Slackにデータを送信するための定義済みのコールバックを公開しています。
```python
from litellm import completion

## ロギングツールの環境変数を設定
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["LLMONITOR_APP_ID"] = "your-llmonitor-app-id"
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"

os.environ["OPENAI_API_KEY"]

# コールバックの設定
litellm.success_callback = ["langfuse", "llmonitor", "athina"] # 入力/出力をLangfuse、LLMonitor、Supabase、Athinaなどにログ

# OpenAIの呼び出し
response = completion(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "こんにちは 👋 - 私はOpenAIです"}])
```

# OpenAIプロキシ（OpenAI Proxy） - ([ドキュメント（Docs）](https://docs.litellm.ai/docs/simple_proxy))

複数のプロジェクトにまたがる予算と料金制限を設定

プロキシが提供するもの: 
1. [認証用フック（Hooks for auth）](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
2. [ロギング用フック（Hooks for logging）](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
3. [コスト追跡（Cost tracking）](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
4. [レート制限（Rate Limiting）](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

## 📖 プロキシエンドポイント（Proxy Endpoints） - [Swaggerドキュメント（Swagger Docs）](https://litellm-api.up.railway.app/)

## クイックスタートプロキシ（Quick Start Proxy） - CLI 

```shell
pip install 'litellm[proxy]'
```

### ステップ1: litellmプロキシを起動
```shell
$ litellm --model huggingface/bigcode/starcoder

#情報（INFO）: プロキシがhttp://0.0.0.0:4000で実行中
```

### ステップ2: プロキシにChatCompletionsリクエストを送信
```python
import openai # openai v1.0.0以降
client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:4000") # プロキシをbase_urlに設定
# litellmプロキシで設定されたモデルに送信されたリクエスト、`litellm --model`
response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
    {
        "role": "user",
        "content": "これはテストリクエストです。短い詩を書いてください。"
    }
])

print(response)
```

## プロキシキー管理（Proxy Key Management） ([ドキュメント（Docs）](https://docs.litellm.ai/docs/proxy/virtual_keys))
プロキシサーバーの`/ui`にあるUI 
![ui_3](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

複数のプロジェクトにまたがる予算と料金制限を設定
`POST /key/generate`

### リクエスト（Request）
```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

### 期待されるレスポンス（Expected Response）
```shell
{
    "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
    "expires": "2023-11-19T01:38:25.838000+00:00" # datetimeオブジェクト
}
```

## サポートされているプロバイダー（Supported Providers） ([ドキュメント（Docs）](https://docs.litellm.ai/docs/providers))
| プロバイダー      | [Completion](https://docs.litellm.ai/docs/#basic-usage) | [Streaming](https://docs.litellm.ai/docs/completion/stream#streaming-responses)  | [Async Completion](https://docs.litellm.ai/docs/completion/stream#async-completion)  | [Async Streaming](https://docs.litellm.ai/docs/completion/stream#async-streaming)  | [Async Embedding](https://docs.litellm.ai/docs/embedding/supported_embedding)  | [Async Image Generation](https://docs.litellm.ai/docs/image_generation)  | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [azure](https://docs.litellm.ai/docs/providers/azure)  | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)  | ✅ | ✅ | ✅ | ✅ |✅ |
| [google - vertex_ai [Gemini]](https://docs.litellm.ai/docs/providers/vertex)  | ✅ | ✅ | ✅ | ✅ |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)  | ✅ | ✅ | ✅ | ✅ |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)  | ✅ |  | ✅ |  | |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅ | ✅ | ✅ | ✅ |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)  | ✅ | ✅ | ✅ | ✅ |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)  | ✅ | ✅ | ✅ | ✅ | ✅ |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)  | ✅ | ✅ | ✅ | ✅ |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)  | ✅ | ✅ | ✅ | ✅ |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)  | ✅ | ✅ | ✅ | ✅ |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)  | ✅ | ✅ | ✅ | ✅ |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)  | ✅ | ✅ | ✅ | ✅ |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)  | ✅ | ✅ | ✅ | ✅ |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)  | ✅ | ✅ | ✅ | ✅ |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)  | ✅ | ✅ | ✅ | ✅ |
| [petals](https://docs.litellm.ai/docs/providers/petals)  | ✅ | ✅ | ✅ | ✅ |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)  | ✅ | ✅ | ✅ | ✅ |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)  | ✅ | ✅ | ✅ | ✅ |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)  | ✅ | ✅ | ✅ | ✅ |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)  | ✅ | ✅ | ✅ | ✅ |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)  | ✅ | ✅ | ✅ | ✅ |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)  |  |  |  |  | ✅ |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference)  |  |  |  |  | ✅ |


[**ドキュメントを読む（Read the Docs）**](https://docs.litellm.ai/docs/)

## 貢献（Contributing）
貢献するには: リポジトリをローカルにクローン → 変更を加える → 変更を含むPRを送信。

リポジトリをローカルで変更する方法は次のとおりです:
ステップ1: リポジトリをクローン
```
git clone https://github.com/BerriAI/litellm.git
```

ステップ2: プロジェクトに移動し、依存関係をインストール:
```
cd litellm
poetry install
```

ステップ3: 変更をテスト:
```
cd litellm/tests # pwd: Documents/litellm/litellm/tests
poetry run flake8
poetry run pytest .
```

ステップ4: 変更を含むPRを送信! 🚀
- フォークしたリポジトリをGitHubにプッシュ
- そこからPRを送信

# エンタープライズ（Enterprise）
より高いセキュリティ、ユーザー管理、プロフェッショナルサポートを必要とする企業向け

[創業者と話す（Talk to founders）](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

これには以下が含まれます:
- ✅ **[LiteLLM商用ライセンス（LiteLLM Commercial License）](https://docs.litellm.ai/docs/proxy/enterprise)の機能:**
- ✅ **機能の優先順位付け（Feature Prioritization）**
- ✅ **カスタム統合（Custom Integrations）**
- ✅ **プロフェッショナルサポート - 専用のDiscord + Slack（Professional Support - Dedicated discord + slack）**
- ✅ **カスタムSLA（Custom SLAs）**
- ✅ **シングルサインオンによる安全なアクセス（Secure access with Single Sign-On）**

# サポート / 創業者と話す
- [デモのスケジュール 👋（Schedule Demo）](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [コミュニティDiscord 💭（Community Discord）](https://discord.gg/wuPM9dRgDw)
- 電話番号 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- メールアドレス ✉️ ishaan@berri.ai / krrish@berri.ai

# なぜ私たちはこれを構築したのか
- **シンプルさの必要性（Need for simplicity）**: Azure、OpenAI、Cohereの間で呼び出しを管理・変換する際、コードが非常に複雑になってきていた。

# 貢献者（Contributors）

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>