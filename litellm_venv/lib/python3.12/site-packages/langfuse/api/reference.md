# Reference
## Comments
<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a comment. Comments may be attached to different object types (trace, observation, session, prompt).
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import CreateCommentRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.comments.create(
    request=CreateCommentRequest(
        project_id="string",
        object_type="string",
        object_id="string",
        content="string",
        author_user_id="string",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateCommentRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all comments
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.comments.get(
    page=1,
    limit=1,
    object_type="string",
    object_id="string",
    author_user_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit
    
</dd>
</dl>

<dl>
<dd>

**object_type:** `typing.Optional[str]` — Filter comments by object type (trace, observation, session, prompt).
    
</dd>
</dl>

<dl>
<dd>

**object_id:** `typing.Optional[str]` — Filter comments by object id. If objectType is not provided, an error will be thrown.
    
</dd>
</dl>

<dl>
<dd>

**author_user_id:** `typing.Optional[str]` — Filter comments by author user id.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.comments.<a href="src/langfuse/resources/comments/client.py">get_by_id</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a comment by id
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.comments.get_by_id(
    comment_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**comment_id:** `str` — The unique langfuse identifier of a comment
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## DatasetItems
<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import CreateDatasetItemRequest, DatasetStatus
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_items.create(
    request=CreateDatasetItemRequest(
        dataset_name="string",
        input={"key": "value"},
        expected_output={"key": "value"},
        metadata={"key": "value"},
        source_trace_id="string",
        source_observation_id="string",
        id="string",
        status=DatasetStatus.ACTIVE,
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_items.get(
    id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.dataset_items.<a href="src/langfuse/resources/dataset_items/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get dataset items
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_items.list(
    dataset_name="string",
    source_trace_id="string",
    source_observation_id="string",
    page=1,
    limit=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**source_trace_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**source_observation_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## DatasetRunItems
<details><summary><code>client.dataset_run_items.<a href="src/langfuse/resources/dataset_run_items/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset run item
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import CreateDatasetRunItemRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.dataset_run_items.create(
    request=CreateDatasetRunItemRequest(
        run_name="string",
        run_description="string",
        metadata={"key": "value"},
        dataset_item_id="string",
        observation_id="string",
        trace_id="string",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetRunItemRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Datasets
<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all datasets
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.list(
    page=1,
    limit=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.get(
    dataset_name="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a dataset
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import CreateDatasetRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.create(
    request=CreateDatasetRequest(
        name="string",
        description="string",
        metadata={"key": "value"},
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateDatasetRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get_run</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a dataset run and its items
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.get_run(
    dataset_name="string",
    run_name="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**run_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.datasets.<a href="src/langfuse/resources/datasets/client.py">get_runs</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get dataset runs
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.datasets.get_runs(
    dataset_name="string",
    page=1,
    limit=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**dataset_name:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Health
<details><summary><code>client.health.<a href="src/langfuse/resources/health/client.py">health</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Check health of API and database
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.health.health()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Ingestion
<details><summary><code>client.ingestion.<a href="src/langfuse/resources/ingestion/client.py">batch</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Batched ingestion for Langfuse Tracing.
If you want to use tracing via the API, such as to build your own Langfuse client implementation, this is the only API route you need to implement.

Within each batch, there can be multiple events.
Each event has a type, an id, a timestamp, metadata and a body.
Internally, we refer to this as the "event envelope" as it tells us something about the event but not the trace.
We use the event id within this envelope to deduplicate messages to avoid processing the same event twice, i.e. the event id should be unique per request.
The event.body.id is the ID of the actual trace and will be used for updates and will be visible within the Langfuse App.
I.e. if you want to update a trace, you'd use the same body id, but separate event IDs.

Notes:

- Introduction to data model: https://langfuse.com/docs/tracing-data-model
- Batch sizes are limited to 3.5 MB in total. You need to adjust the number of events per batch accordingly.
- The API does not return a 4xx status code for input errors. Instead, it responds with a 207 status code, which includes a list of the encountered errors.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import IngestionEvent_ScoreCreate, ScoreBody
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.ingestion.batch(
    batch=[
        IngestionEvent_ScoreCreate(
            id="abcdef-1234-5678-90ab",
            timestamp="2022-01-01T00:00:00.000Z",
            body=ScoreBody(
                id="abcdef-1234-5678-90ab",
                trace_id="1234-5678-90ab-cdef",
                name="My Score",
                value=0.9,
            ),
        )
    ],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**batch:** `typing.Sequence[IngestionEvent]` — Batch of tracing events to be ingested. Discriminated by attribute `type`.
    
</dd>
</dl>

<dl>
<dd>

**metadata:** `typing.Optional[typing.Any]` — Optional. Metadata field used by the Langfuse SDKs for debugging.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Media
<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.media.get(
    media_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**media_id:** `str` — The unique langfuse identifier of a media record
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">patch</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Patch a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse import PatchMediaBody
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.media.patch(
    media_id="string",
    request=PatchMediaBody(
        uploaded_at=datetime.datetime.fromisoformat(
            "2024-01-15 09:30:00+00:00",
        ),
        upload_http_status=1,
        upload_http_error="string",
        upload_time_ms=1,
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**media_id:** `str` — The unique langfuse identifier of a media record
    
</dd>
</dl>

<dl>
<dd>

**request:** `PatchMediaBody` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.media.<a href="src/langfuse/resources/media/client.py">get_upload_url</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a presigned upload URL for a media record
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import GetMediaUploadUrlRequest, MediaContentType
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.media.get_upload_url(
    request=GetMediaUploadUrlRequest(
        trace_id="string",
        observation_id="string",
        content_type=MediaContentType.IMAGE_PNG,
        content_length=1,
        sha_256_hash="string",
        field="string",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `GetMediaUploadUrlRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Metrics
<details><summary><code>client.metrics.<a href="src/langfuse/resources/metrics/client.py">daily</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get daily metrics of the Langfuse project
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.metrics.daily(
    page=1,
    limit=1,
    trace_name="string",
    user_id="string",
    tags="string",
    from_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**trace_name:** `typing.Optional[str]` — Optional filter by the name of the trace
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` — Optional filter by the userId associated with the trace
    
</dd>
</dl>

<dl>
<dd>

**tags:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Optional filter for metrics where traces include all of these tags
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces and observations on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces and observations before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Models
<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a model
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse import CreateModelRequest, ModelUsageUnit
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.models.create(
    request=CreateModelRequest(
        model_name="string",
        match_pattern="string",
        start_date=datetime.datetime.fromisoformat(
            "2024-01-15 09:30:00+00:00",
        ),
        unit=ModelUsageUnit.CHARACTERS,
        input_price=1.1,
        output_price=1.1,
        total_price=1.1,
        tokenizer_id="string",
        tokenizer_config={"key": "value"},
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateModelRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all models
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.models.list(
    page=1,
    limit=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a model
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.models.get(
    id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.models.<a href="src/langfuse/resources/models/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a model. Cannot delete models managed by Langfuse. You can create your own definition with the same modelName to override the definition though.
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.models.delete(
    id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**id:** `str` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Observations
<details><summary><code>client.observations.<a href="src/langfuse/resources/observations/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a observation
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.observations.get(
    observation_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**observation_id:** `str` — The unique langfuse identifier of an observation, can be an event, span or generation
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.observations.<a href="src/langfuse/resources/observations/client.py">get_many</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of observations
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.observations.get_many(
    page=1,
    limit=1,
    name="string",
    user_id="string",
    type="string",
    trace_id="string",
    parent_observation_id="string",
    from_start_time=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_start_time=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    version="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**type:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**trace_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**parent_observation_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**from_start_time:** `typing.Optional[dt.datetime]` — Retrieve only observations with a start_time or or after this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**to_start_time:** `typing.Optional[dt.datetime]` — Retrieve only observations with a start_time before this datetime (ISO 8601).
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[str]` — Optional filter to only include observations with a certain version.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Projects
<details><summary><code>client.projects.<a href="src/langfuse/resources/projects/client.py">get</a>()</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get Project associated with API key
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.projects.get()

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## PromptVersion
<details><summary><code>client.prompt_version.<a href="src/langfuse/resources/prompt_version/client.py">update</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Update labels for a specific prompt version
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompt_version.update(
    name="string",
    version=1,
    new_labels=["string"],
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `str` — The name of the prompt
    
</dd>
</dl>

<dl>
<dd>

**version:** `int` — Version of the prompt to update
    
</dd>
</dl>

<dl>
<dd>

**new_labels:** `typing.Sequence[str]` — New labels for the prompt version. Labels are unique across versions. The "latest" label is reserved and managed by Langfuse.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Prompts
<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a prompt
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompts.get(
    prompt_name="string",
    version=1,
    label="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**prompt_name:** `str` — The name of the prompt
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[int]` — Version of the prompt to be retrieved.
    
</dd>
</dl>

<dl>
<dd>

**label:** `typing.Optional[str]` — Label of the prompt to be retrieved. Defaults to "production" if no label or version is set.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of prompt names with versions and labels
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompts.list(
    name="string",
    label="string",
    tag="string",
    page=1,
    limit=1,
    from_updated_at=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_updated_at=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**label:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**tag:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**page:** `typing.Optional[int]` — page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — limit of items per page
    
</dd>
</dl>

<dl>
<dd>

**from_updated_at:** `typing.Optional[dt.datetime]` — Optional filter to only include prompt versions created/updated on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_updated_at:** `typing.Optional[dt.datetime]` — Optional filter to only include prompt versions created/updated before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.prompts.<a href="src/langfuse/resources/prompts/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a new version for the prompt with the given `name`
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import ChatMessage, CreatePromptRequest_Chat
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.prompts.create(
    request=CreatePromptRequest_Chat(
        name="string",
        prompt=[
            ChatMessage(
                role="string",
                content="string",
            )
        ],
        config={"key": "value"},
        labels=["string"],
        tags=["string"],
        commit_message="string",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreatePromptRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## ScoreConfigs
<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a score configuration (config). Score configs are used to define the structure of scores
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import ConfigCategory, CreateScoreConfigRequest, ScoreDataType
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_configs.create(
    request=CreateScoreConfigRequest(
        name="string",
        data_type=ScoreDataType.NUMERIC,
        categories=[
            ConfigCategory(
                value=1.1,
                label="string",
            )
        ],
        min_value=1.1,
        max_value=1.1,
        description="string",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateScoreConfigRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get all score configs
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_configs.get(
    page=1,
    limit=1,
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score_configs.<a href="src/langfuse/resources/score_configs/client.py">get_by_id</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a score config
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score_configs.get_by_id(
    config_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**config_id:** `str` — The unique langfuse identifier of a score config
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Score
<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">create</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Create a score
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse import CreateScoreRequest
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.create(
    request=CreateScoreRequest(
        name="novelty",
        value=0.9,
        trace_id="cdef-1234-5678-90ab",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**request:** `CreateScoreRequest` 
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a list of scores
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse import ScoreDataType, ScoreSource
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.get(
    page=1,
    limit=1,
    user_id="string",
    name="string",
    from_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    source=ScoreSource.ANNOTATION,
    operator="string",
    value=1.1,
    score_ids="string",
    config_id="string",
    queue_id="string",
    data_type=ScoreDataType.NUMERIC,
    trace_tags="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1.
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` — Retrieve only scores with this userId associated to the trace.
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` — Retrieve only scores with this name.
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include scores created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include scores created before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**source:** `typing.Optional[ScoreSource]` — Retrieve only scores from a specific source.
    
</dd>
</dl>

<dl>
<dd>

**operator:** `typing.Optional[str]` — Retrieve only scores with <operator> value.
    
</dd>
</dl>

<dl>
<dd>

**value:** `typing.Optional[float]` — Retrieve only scores with <operator> value.
    
</dd>
</dl>

<dl>
<dd>

**score_ids:** `typing.Optional[str]` — Comma-separated list of score IDs to limit the results to.
    
</dd>
</dl>

<dl>
<dd>

**config_id:** `typing.Optional[str]` — Retrieve only scores with a specific configId.
    
</dd>
</dl>

<dl>
<dd>

**queue_id:** `typing.Optional[str]` — Retrieve only scores with a specific annotation queueId.
    
</dd>
</dl>

<dl>
<dd>

**data_type:** `typing.Optional[ScoreDataType]` — Retrieve only scores with a specific dataType.
    
</dd>
</dl>

<dl>
<dd>

**trace_tags:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Only scores linked to traces that include all of these tags will be returned.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">get_by_id</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a score
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.get_by_id(
    score_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**score_id:** `str` — The unique langfuse identifier of a score
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.score.<a href="src/langfuse/resources/score/client.py">delete</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Delete a score
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.score.delete(
    score_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**score_id:** `str` — The unique langfuse identifier of a score
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Sessions
<details><summary><code>client.sessions.<a href="src/langfuse/resources/sessions/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get sessions
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.sessions.list(
    page=1,
    limit=1,
    from_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include sessions created on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include sessions created before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.sessions.<a href="src/langfuse/resources/sessions/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a session. Please note that `traces` on this endpoint are not paginated, if you plan to fetch large sessions, consider `GET /api/public/traces?sessionId=<sessionId>`
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.sessions.get(
    session_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**session_id:** `str` — The unique id of a session
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

## Trace
<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">get</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get a specific trace
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.trace.get(
    trace_id="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**trace_id:** `str` — The unique langfuse identifier of a trace
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

<details><summary><code>client.trace.<a href="src/langfuse/resources/trace/client.py">list</a>(...)</code></summary>
<dl>
<dd>

#### 📝 Description

<dl>
<dd>

<dl>
<dd>

Get list of traces
</dd>
</dl>
</dd>
</dl>

#### 🔌 Usage

<dl>
<dd>

<dl>
<dd>

```python
import datetime

from langfuse.client import FernLangfuse

client = FernLangfuse(
    x_langfuse_sdk_name="YOUR_X_LANGFUSE_SDK_NAME",
    x_langfuse_sdk_version="YOUR_X_LANGFUSE_SDK_VERSION",
    x_langfuse_public_key="YOUR_X_LANGFUSE_PUBLIC_KEY",
    username="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    base_url="https://yourhost.com/path/to/api",
)
client.trace.list(
    page=1,
    limit=1,
    user_id="string",
    name="string",
    session_id="string",
    from_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    to_timestamp=datetime.datetime.fromisoformat(
        "2024-01-15 09:30:00+00:00",
    ),
    order_by="string",
    tags="string",
    version="string",
    release="string",
)

```
</dd>
</dl>
</dd>
</dl>

#### ⚙️ Parameters

<dl>
<dd>

<dl>
<dd>

**page:** `typing.Optional[int]` — Page number, starts at 1
    
</dd>
</dl>

<dl>
<dd>

**limit:** `typing.Optional[int]` — Limit of items per page. If you encounter api issues due to too large page sizes, try to reduce the limit.
    
</dd>
</dl>

<dl>
<dd>

**user_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**name:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**session_id:** `typing.Optional[str]` 
    
</dd>
</dl>

<dl>
<dd>

**from_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces with a trace.timestamp on or after a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**to_timestamp:** `typing.Optional[dt.datetime]` — Optional filter to only include traces with a trace.timestamp before a certain datetime (ISO 8601)
    
</dd>
</dl>

<dl>
<dd>

**order_by:** `typing.Optional[str]` — Format of the string [field].[asc/desc]. Fields: id, timestamp, name, userId, release, version, public, bookmarked, sessionId. Example: timestamp.asc
    
</dd>
</dl>

<dl>
<dd>

**tags:** `typing.Optional[typing.Union[str, typing.Sequence[str]]]` — Only traces that include all of these tags will be returned.
    
</dd>
</dl>

<dl>
<dd>

**version:** `typing.Optional[str]` — Optional filter to only include traces with a certain version.
    
</dd>
</dl>

<dl>
<dd>

**release:** `typing.Optional[str]` — Optional filter to only include traces with a certain release.
    
</dd>
</dl>

<dl>
<dd>

**request_options:** `typing.Optional[RequestOptions]` — Request-specific configuration.
    
</dd>
</dl>
</dd>
</dl>


</dd>
</dl>
</details>

