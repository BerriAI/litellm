## 核心 API

### 对话补全

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/deployments/{model}/chat/completions` | 对话补全 |
| `POST` | `/engines/{model}/chat/completions` | 对话补全 |
| `POST` | `/chat/completions` | 对话补全 |
| `POST` | `/v1/chat/completions` | 对话补全 |

#### `POST` /openai/deployments/{model}/chat/completions

**对话补全**

与 OpenAI Chat API 完全兼容：https://platform.openai.com/docs/api-reference/chat

```bash
curl -X POST http://localhost:4000/v1/chat/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /engines/{model}/chat/completions

**对话补全**

与 OpenAI Chat API 完全兼容：https://platform.openai.com/docs/api-reference/chat

```bash
curl -X POST http://localhost:4000/v1/chat/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /chat/completions

**对话补全**

与 OpenAI Chat API 完全兼容：https://platform.openai.com/docs/api-reference/chat

```bash
curl -X POST http://localhost:4000/v1/chat/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}'
```

#### `POST` /v1/chat/completions

**对话补全**

与 OpenAI Chat API 完全兼容：https://platform.openai.com/docs/api-reference/chat

```bash
curl -X POST http://localhost:4000/v1/chat/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
}'
```

### 文本补全

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/deployments/{model}/completions` | 文本补全 |
| `POST` | `/engines/{model}/completions` | 文本补全 |
| `POST` | `/completions` | 文本补全 |
| `POST` | `/v1/completions` | 文本补全 |

#### `POST` /openai/deployments/{model}/completions

**文本补全**

与 OpenAI Completions API 完全兼容：https://platform.openai.com/docs/api-reference/completions

```bash
curl -X POST http://localhost:4000/v1/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /engines/{model}/completions

**文本补全**

与 OpenAI Completions API 完全兼容：https://platform.openai.com/docs/api-reference/completions

```bash
curl -X POST http://localhost:4000/v1/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /completions

**文本补全**

与 OpenAI Completions API 完全兼容：https://platform.openai.com/docs/api-reference/completions

```bash
curl -X POST http://localhost:4000/v1/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

#### `POST` /v1/completions

**文本补全**

与 OpenAI Completions API 完全兼容：https://platform.openai.com/docs/api-reference/completions

```bash
curl -X POST http://localhost:4000/v1/completions 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "gpt-3.5-turbo-instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 0.7
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

### 嵌入向量

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/deployments/{model}/embeddings` | 嵌入向量 |
| `POST` | `/engines/{model}/embeddings` | 嵌入向量 |
| `POST` | `/embeddings` | 嵌入向量 |
| `POST` | `/v1/embeddings` | 嵌入向量 |

#### `POST` /openai/deployments/{model}/embeddings

**嵌入向量**

与 OpenAI Embeddings API 完全兼容：https://platform.openai.com/docs/api-reference/embeddings

```bash
curl -X POST http://localhost:4000/v1/embeddings 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /engines/{model}/embeddings

**嵌入向量**

与 OpenAI Embeddings API 完全兼容：https://platform.openai.com/docs/api-reference/embeddings

```bash
curl -X POST http://localhost:4000/v1/embeddings 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
}'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /embeddings

**嵌入向量**

与 OpenAI Embeddings API 完全兼容：https://platform.openai.com/docs/api-reference/embeddings

```bash
curl -X POST http://localhost:4000/v1/embeddings 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
}'
```

#### `POST` /v1/embeddings

**嵌入向量**

与 OpenAI Embeddings API 完全兼容：https://platform.openai.com/docs/api-reference/embeddings

```bash
curl -X POST http://localhost:4000/v1/embeddings 
-H "Content-Type: application/json" 
-H "Authorization: Bearer sk-1234" 
-d '{
    "model": "text-embedding-ada-002",
    "input": "The quick brown fox jumps over the lazy dog"
}'
```

### 内容审核

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/moderations` | 内容审核 |
| `POST` | `/v1/moderations` | 内容审核 |

#### `POST` /moderations

**内容审核**

Moderations 端点用于检查内容是否符合 LLM 提供商的策略要求。
快速开始
```
curl --location 'http://0.0.0.0:4000/moderations'     --header 'Content-Type: application/json'     --header 'Authorization: Bearer sk-1234'     --data '{"input": "Sample text goes here", "model": "text-moderation-stable"}'
```

#### `POST` /v1/moderations

**内容审核**

Moderations 端点用于检查内容是否符合 LLM 提供商的策略要求。
快速开始
```
curl --location 'http://0.0.0.0:4000/moderations'     --header 'Content-Type: application/json'     --header 'Authorization: Bearer sk-1234'     --data '{"input": "Sample text goes here", "model": "text-moderation-stable"}'
```

### 音频

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/audio/speech` | 语音合成 |
| `POST` | `/v1/audio/speech` | 语音合成 |
| `POST` | `/audio/transcriptions` | 语音转文字 |
| `POST` | `/v1/audio/transcriptions` | 语音转文字 |

#### `POST` /audio/speech

**语音合成**

参数与下列规范相同：

https://platform.openai.com/docs/api-reference/audio/createSpeech

#### `POST` /v1/audio/speech

**语音合成**

参数与下列规范相同：

https://platform.openai.com/docs/api-reference/audio/createSpeech

#### `POST` /audio/transcriptions

**语音转文字**

参数与下列规范相同：

https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=curl

#### `POST` /v1/audio/transcriptions

**语音转文字**

参数与下列规范相同：

https://platform.openai.com/docs/api-reference/audio/createTranscription?lang=curl

### 图片

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/deployments/{model}/images/generations` | 图片生成 |
| `POST` | `/images/generations` | 图片生成 |
| `POST` | `/v1/images/generations` | 图片生成 |
| `POST` | `/openai/deployments/{model}/images/edits` | Image Edit Api |
| `POST` | `/images/edits` | Image Edit Api |
| `POST` | `/v1/images/edits` | Image Edit Api |

#### `POST` /openai/deployments/{model}/images/generations

**图片生成**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /images/generations

**图片生成**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

#### `POST` /v1/images/generations

**图片生成**

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

#### `POST` /openai/deployments/{model}/images/edits

**图片编辑 API**

与 OpenAI Images API 规范一致：https://platform.openai.com/docs/api-reference/images/create

```bash
curl -s -D >(grep -i x-request-id >&2)     -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png)     -X POST "http://localhost:4000/v1/images/edits"     -H "Authorization: Bearer sk-1234"         -F "model=gpt-image-1"         -F "image[]=@soap.png"         -F 'prompt=Create a studio ghibli image of this'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 路径参数 | ✅ |  |

#### `POST` /images/edits

**图片编辑 API**

与 OpenAI Images API 规范一致：https://platform.openai.com/docs/api-reference/images/create

```bash
curl -s -D >(grep -i x-request-id >&2)     -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png)     -X POST "http://localhost:4000/v1/images/edits"     -H "Authorization: Bearer sk-1234"         -F "model=gpt-image-1"         -F "image[]=@soap.png"         -F 'prompt=Create a studio ghibli image of this'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

#### `POST` /v1/images/edits

**图片编辑 API**

与 OpenAI Images API 规范一致：https://platform.openai.com/docs/api-reference/images/create

```bash
curl -s -D >(grep -i x-request-id >&2)     -o >(jq -r '.data[0].b64_json' | base64 --decode > gift-basket.png)     -X POST "http://localhost:4000/v1/images/edits"     -H "Authorization: Bearer sk-1234"         -F "model=gpt-image-1"         -F "image[]=@soap.png"         -F 'prompt=Create a studio ghibli image of this'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `model` | 查询参数 | ❌ |  |

### 视频

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/videos` | Video List |
| `POST` | `/videos` | Video Generation |
| `GET` | `/v1/videos` | Video List |
| `POST` | `/v1/videos` | Video Generation |
| `GET` | `/videos/{video_id}` | Video Status |
| `GET` | `/v1/videos/{video_id}` | Video Status |
| `GET` | `/videos/{video_id}/content` | Video Content |
| `GET` | `/v1/videos/{video_id}/content` | Video Content |
| `POST` | `/videos/{video_id}/remix` | Video Remix |
| `POST` | `/v1/videos/{video_id}/remix` | Video Remix |
| `POST` | `/videos/characters` | Video Create Character |
| `POST` | `/v1/videos/characters` | Video Create Character |
| `GET` | `/videos/characters/{character_id}` | Video Get Character |
| `GET` | `/v1/videos/characters/{character_id}` | Video Get Character |
| `POST` | `/videos/edits` | Video Edit |
| `POST` | `/v1/videos/edits` | Video Edit |
| `POST` | `/videos/extensions` | Video Extension |
| `POST` | `/v1/videos/extensions` | Video Extension |

#### `GET` /videos

**视频列表**

视频列表端点，用于获取视频列表。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos"         -H "Authorization: Bearer sk-1234"
```

#### `POST` /videos

**视频生成**

视频生成端点，根据文本提示词生成视频。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model": "sora-2",
        "prompt": "A beautiful sunset over the ocean"
    }'
```

#### `GET` /v1/videos

**视频列表**

视频列表端点，用于获取视频列表。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos"         -H "Authorization: Bearer sk-1234"
```

#### `POST` /v1/videos

**视频生成**

视频生成端点，根据文本提示词生成视频。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model": "sora-2",
        "prompt": "A beautiful sunset over the ocean"
    }'
```

#### `GET` /videos/{video_id}

**视频状态**

视频状态端点，用于查询视频状态与元数据。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/video_123"         -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `GET` /v1/videos/{video_id}

**视频状态**

视频状态端点，用于查询视频状态与元数据。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/video_123"         -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `GET` /videos/{video_id}/content

**视频内容**

视频内容端点，用于下载视频内容。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/{video_id}/content"         -H "Authorization: Bearer sk-1234"         --output video.mp4
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `GET` /v1/videos/{video_id}/content

**视频内容**

视频内容端点，用于下载视频内容。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/{video_id}/content"         -H "Authorization: Bearer sk-1234"         --output video.mp4
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `POST` /videos/{video_id}/remix

**视频重混**

视频重混端点，使用新的提示词对已有视频进行重混。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/video_123/remix"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "prompt": "A new version with different colors"
    }'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `POST` /v1/videos/{video_id}/remix

**视频重混**

视频重混端点，使用新的提示词对已有视频进行重混。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/video_123/remix"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "prompt": "A new version with different colors"
    }'
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `video_id` | 路径参数 | ✅ |  |

#### `POST` /videos/characters

**创建 Character**

从上传的视频文件创建 Character。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-character

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/characters"         -H "Authorization: Bearer sk-1234"         -F "video=@character_video.mp4"         -F "name=my_character"
```

#### `POST` /v1/videos/characters

**创建 Character**

从上传的视频文件创建 Character。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-character

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/characters"         -H "Authorization: Bearer sk-1234"         -F "video=@character_video.mp4"         -F "name=my_character"
```

#### `GET` /videos/characters/{character_id}

**获取 Character**

根据 ID 获取 Character。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/get-character

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/characters/char_123"         -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `character_id` | 路径参数 | ✅ |  |

#### `GET` /v1/videos/characters/{character_id}

**获取 Character**

根据 ID 获取 Character。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/get-character

示例：
```bash
curl -X GET "http://localhost:4000/v1/videos/characters/char_123"         -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `character_id` | 路径参数 | ✅ |  |

#### `POST` /videos/edits

**视频编辑**

创建视频编辑作业。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-edit

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/edits"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{"prompt": "Make it brighter", "video": {"id": "video_123"}}'
```

#### `POST` /v1/videos/edits

**视频编辑**

创建视频编辑作业。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-edit

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/edits"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{"prompt": "Make it brighter", "video": {"id": "video_123"}}'
```

#### `POST` /videos/extensions

**视频续拍**

创建视频续拍任务。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-extension

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/extensions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{"prompt": "Continue the scene", "seconds": "5", "video": {"id": "video_123"}}'
```

#### `POST` /v1/videos/extensions

**视频续拍**

创建视频续拍任务。

与 OpenAI Videos API 规范一致：
https://platform.openai.com/docs/api-reference/videos/create-extension

示例：
```bash
curl -X POST "http://localhost:4000/v1/videos/extensions"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{"prompt": "Continue the scene", "seconds": "5", "video": {"id": "video_123"}}'
```

### OCR 文字识别

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/ocr` | OCR 文字识别 |
| `POST` | `/v1/ocr` | OCR 文字识别 |

#### `POST` /ocr

**OCR 文字识别**

OCR 端点，用于从文档与图片中提取文本。

支持两种输入模式：

**1. JSON body** (Mistral OCR API compatible):
```bash
curl -X POST "http://localhost:4000/v1/ocr"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model": "mistral-ocr",
        "document": {
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2201.04234"
        }
    }'
```

**2. Multipart form file upload**:
``...

#### `POST` /v1/ocr

**OCR 文字识别**

OCR 端点，用于从文档与图片中提取文本。

支持两种输入模式：

**1. JSON body** (Mistral OCR API compatible):
```bash
curl -X POST "http://localhost:4000/v1/ocr"         -H "Authorization: Bearer sk-1234"         -H "Content-Type: application/json"         -d '{
        "model": "mistral-ocr",
        "document": {
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2201.04234"
        }
    }'
```

**2. Multipart form file upload**:
``...

### 重排序

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/rerank` | 重排序 |
| `POST` | `/v1/rerank` | 重排序 |
| `POST` | `/v2/rerank` | 重排序 |

### 响应

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/v1/responses` | Responses Api |
| `POST` | `/cursor/chat/completions` | Cursor Chat Completions |
| `GET` | `/openai/v1/responses/{response_id}` | 获取响应 |
| `DELETE` | `/openai/v1/responses/{response_id}` | 删除响应 |
| `GET` | `/responses/{response_id}` | 获取响应 |
| `DELETE` | `/responses/{response_id}` | 删除响应 |
| `GET` | `/v1/responses/{response_id}` | 获取响应 |
| `DELETE` | `/v1/responses/{response_id}` | 删除响应 |
| `GET` | `/openai/v1/responses/{response_id}/input_items` | 获取 Response Input Items |
| `GET` | `/responses/{response_id}/input_items` | 获取 Response Input Items |
| `GET` | `/v1/responses/{response_id}/input_items` | 获取 Response Input Items |
| `POST` | `/openai/v1/responses/compact` | Compact Response |
| `POST` | `/responses/compact` | Compact Response |
| `POST` | `/v1/responses/compact` | Compact Response |
| `POST` | `/openai/v1/responses/{response_id}/cancel` | 取消 Response |
| `POST` | `/responses/{response_id}/cancel` | 取消 Response |
| `POST` | `/v1/responses/{response_id}/cancel` | 取消 Response |

#### `POST` /openai/v1/responses

**Responses API**

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses

支持 `polling_via_cache` 的后台模式，用于轮询获取部分响应。
当 `background=true` 且启用了 `polling_via_cache` 时，立即返回 `polling_id`。
并在后台流式将响应写入 Redis 缓存。

```bash
# Normal request
curl -X POST http://localhost:4000/v1/responses     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"     -d '{
    "m...
```

#### `POST` /cursor/chat/completions

**Cursor Chat Completions**

专为 Cursor 设计的端点：接收 Responses API 的输入格式，返回 Chat Completions 格式。

该端点处理来自 Cursor IDE 的请求：请求采用 Responses API 格式（`input` 字段），但期望返回 Chat Completions 格式的响应（包含 `choices`、`messages` 等）。

```bash
curl -X POST http://localhost:4000/cursor/chat/completions     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"     -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "He...
```

#### `GET` /openai/v1/responses/{response_id}

**获取响应**

根据 ID 获取 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/get

```bash
# Get polling response
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123     -H "Authorization: Bearer sk-1234"

# Get provider response
curl -X GET http://localhost:4000/v1/responses/res...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `DELETE` /openai/v1/responses/{response_id}

**删除响应**

根据 ID 删除 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Deletes from Redis cache
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/delete

```bash
curl -X DELETE http://localhost:4000/v1/responses/resp_abc123     -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `GET` /responses/{response_id}

**获取响应**

根据 ID 获取 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/get

```bash
# Get polling response
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123     -H "Authorization: Bearer sk-1234"

# Get provider response
curl -X GET http://localhost:4000/v1/responses/res...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `DELETE` /responses/{response_id}

**删除响应**

根据 ID 删除 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Deletes from Redis cache
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/delete

```bash
curl -X DELETE http://localhost:4000/v1/responses/resp_abc123     -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `GET` /v1/responses/{response_id}

**获取响应**

根据 ID 获取 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Returns cumulative cached content from background responses
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/get

```bash
# Get polling response
curl -X GET http://localhost:4000/v1/responses/litellm_poll_abc123     -H "Authorization: Bearer sk-1234"

# Get provider response
curl -X GET http://localhost:4000/v1/responses/res...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `DELETE` /v1/responses/{response_id}

**删除响应**

根据 ID 删除 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Deletes from Redis cache
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/delete

```bash
curl -X DELETE http://localhost:4000/v1/responses/resp_abc123     -H "Authorization: Bearer sk-1234"
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `GET` /openai/v1/responses/{response_id}/input_items

**获取 Response Input Items**

列出 Response 的输入条目。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `GET` /responses/{response_id}/input_items

**获取 Response Input Items**

列出 Response 的输入条目。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `GET` /v1/responses/{response_id}/input_items

**获取 Response Input Items**

列出 Response 的输入条目。

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `POST` /openai/v1/responses/compact

**压缩 Response**

通过对会话执行压缩流程来压缩 Response。

返回加密的不透明条目，可用于压缩上下文长度。

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/compact

```bash
curl -X POST http://localhost:4000/v1/responses/compact     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"     -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "Hello"}]
}'
```

#### `POST` /responses/compact

**压缩 Response**

通过对会话执行压缩流程来压缩 Response。

返回加密的不透明条目，可用于压缩上下文长度。

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/compact

```bash
curl -X POST http://localhost:4000/v1/responses/compact     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"     -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "Hello"}]
}'
```

#### `POST` /v1/responses/compact

**压缩 Response**

通过对会话执行压缩流程来压缩 Response。

返回加密的不透明条目，可用于压缩上下文长度。

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/compact

```bash
curl -X POST http://localhost:4000/v1/responses/compact     -H "Content-Type: application/json"     -H "Authorization: Bearer sk-1234"     -d '{
    "model": "gpt-4o",
    "input": [{"role": "user", "content": "Hello"}]
}'
```

#### `POST` /openai/v1/responses/{response_id}/cancel

**取消 Response**

根据 ID 取消 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/cancel

```bash
# Cancel polling response
curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel     -H "Authorization: Bearer sk-1234"

# Cancel provider response
curl -X POST http://localhost:4000...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `POST` /responses/{response_id}/cancel

**取消 Response**

根据 ID 取消 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/cancel

```bash
# Cancel polling response
curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel     -H "Authorization: Bearer sk-1234"

# Cancel provider response
curl -X POST http://localhost:4000...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

#### `POST` /v1/responses/{response_id}/cancel

**取消 Response**

根据 ID 取消 Response。

两种方式均支持：
- Polling IDs (litellm_poll_*): Cancels background response and updates status in Redis
- Provider response IDs: Passes through to provider API

与 OpenAI Responses API 规范一致：https://platform.openai.com/docs/api-reference/responses/cancel

```bash
# Cancel polling response
curl -X POST http://localhost:4000/v1/responses/litellm_poll_abc123/cancel     -H "Authorization: Bearer sk-1234"

# Cancel provider response
curl -X POST http://localhost:4000...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `response_id` | 路径参数 | ✅ |  |

### 实时

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/openai/v1/realtime/client_secrets` | 创建 Realtime Client Secret |
| `POST` | `/realtime/client_secrets` | 创建 Realtime Client Secret |
| `POST` | `/v1/realtime/client_secrets` | 创建 Realtime Client Secret |
| `POST` | `/openai/v1/realtime/calls` | Proxy Realtime Calls |
| `POST` | `/realtime/calls` | Proxy Realtime Calls |
| `POST` | `/v1/realtime/calls` | Proxy Realtime Calls |

### 搜索

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/search` | Search |
| `POST` | `/v1/search` | Search |
| `POST` | `/search/{search_tool_name}` | Search |
| `POST` | `/v1/search/{search_tool_name}` | Search |
| `GET` | `/search/tools` | 列表 Search Tools |
| `GET` | `/v1/search/tools` | 列表 Search Tools |

#### `POST` /search

**搜索**

搜索端点，用于执行 Web 搜索。

与 Perplexity Search API 规范一致：
https://docs.perplexity.ai/api-reference/search-post

`search_tool_name` 可通过以下任一方式传入：
1. In the URL path: /v1/search/{search_tool_name}
2. In the request body: {"search_tool_name": "..."}

在 URL 中带上 `search_tool_name` 的示例（推荐方式，可保持请求体与 Perplexity 兼容）：
```bash
curl -X POST "http://localhost:4000/v1/search/litellm-search"         -H "Authorization: Bearer sk-1234"         -H "Co...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_name` | 查询参数 | ❌ |  |

#### `POST` /v1/search

**搜索**

搜索端点，用于执行 Web 搜索。

与 Perplexity Search API 规范一致：
https://docs.perplexity.ai/api-reference/search-post

`search_tool_name` 可通过以下任一方式传入：
1. In the URL path: /v1/search/{search_tool_name}
2. In the request body: {"search_tool_name": "..."}

在 URL 中带上 `search_tool_name` 的示例（推荐方式，可保持请求体与 Perplexity 兼容）：
```bash
curl -X POST "http://localhost:4000/v1/search/litellm-search"         -H "Authorization: Bearer sk-1234"         -H "Co...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_name` | 查询参数 | ❌ |  |

#### `POST` /search/{search_tool_name}

**搜索**

搜索端点，用于执行 Web 搜索。

与 Perplexity Search API 规范一致：
https://docs.perplexity.ai/api-reference/search-post

`search_tool_name` 可通过以下任一方式传入：
1. In the URL path: /v1/search/{search_tool_name}
2. In the request body: {"search_tool_name": "..."}

在 URL 中带上 `search_tool_name` 的示例（推荐方式，可保持请求体与 Perplexity 兼容）：
```bash
curl -X POST "http://localhost:4000/v1/search/litellm-search"         -H "Authorization: Bearer sk-1234"         -H "Co...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_name` | 路径参数 | ✅ |  |

#### `POST` /v1/search/{search_tool_name}

**搜索**

搜索端点，用于执行 Web 搜索。

与 Perplexity Search API 规范一致：
https://docs.perplexity.ai/api-reference/search-post

`search_tool_name` 可通过以下任一方式传入：
1. In the URL path: /v1/search/{search_tool_name}
2. In the request body: {"search_tool_name": "..."}

在 URL 中带上 `search_tool_name` 的示例（推荐方式，可保持请求体与 Perplexity 兼容）：
```bash
curl -X POST "http://localhost:4000/v1/search/litellm-search"         -H "Authorization: Bearer sk-1234"         -H "Co...
```

**参数：**

| 参数名 | 位置 | 必填 | 说明 |
|--------|------|------|------|
| `search_tool_name` | 路径参数 | ✅ |  |

#### `GET` /search/tools

**列表 Search Tools**

列出路由中配置的全部可用搜索工具。

该端点返回当前已加载且可用的搜索工具。
供 `/v1/search` 端点使用。

示例：
```bash
curl -X GET "http://localhost:4000/v1/search/tools"         -H "Authorization: Bearer sk-1234"
```

响应：
```
{
    "object": "list",
    "data": [
        {
            "search_tool_name": "litellm-search",
            "search_provider": "perplexity",
            "description": "Perplexity search...
```

#### `GET` /v1/search/tools

**列表 Search Tools**

列出路由中配置的全部可用搜索工具。

该端点返回当前已加载且可用的搜索工具。
供 `/v1/search` 端点使用。

示例：
```bash
curl -X GET "http://localhost:4000/v1/search/tools"         -H "Authorization: Bearer sk-1234"
```

响应：
```
{
    "object": "list",
    "data": [
        {
            "search_tool_name": "litellm-search",
            "search_provider": "perplexity",
            "description": "Perplexity search...
```

