# Gnani batch speech-to-text

Prisma v2.5 and v3.0 can transcribe long recordings and multiple files in one job. Gnani accepts recordings up to four hours per file. Use a public HTTPS audio URL for large files, or upload local files as multipart data. Gnani documents a 10 MB limit per directly uploaded audio file. [Batch limits](https://docs.gnani.ai/api/STTBatch/Introduction#limits)

## Gateway

Set `GNANI_API_KEY` on the proxy. Alternatively, configure a proxy-wide Gnani deployment with `use_in_pass_through: true`. These routes use a proxy admin key because the provider credential can list, read and cancel jobs throughout its Gnani organization. Gnani bills that account for processing. LiteLLM's OpenAI-compatible JSONL batches use a separate API.

```bash
jobs="http://localhost:4000/gnani/stt/v3/batch/jobs"
auth="Authorization: Bearer $LITELLM_MASTER_KEY"
job_id=$(curl --fail-with-body -sS "$jobs" -H "$auth" \
  -F 'config={"model":"prisma-v3.0","language_code":"en-IN","with_diarization":true,"num_speakers":2};type=application/json' \
  -F 'files=@meeting.wav' | jq -er .job_id)
curl --fail-with-body -sS -X POST "$jobs/$job_id/start" -H "$auth"
curl --fail-with-body -sS "$jobs/$job_id" -H "$auth"
```

Repeat the `files` field to submit several recordings. Add `-F "callback_url=$CALLBACK_URL"` at creation to receive Gnani's terminal notifications at an HTTPS endpoint you control. The callback includes the job ID, the terminal status and any completed transcripts. Delivery is best effort. Poll at intervals of at least ten seconds for the authoritative status. For a completed or partially successful job, inspect each file.

```bash
curl --fail-with-body -sS "$jobs/$job_id/files" -H "$auth" > files.json
url=$(jq -er 'first(.data[] | select(.status == "COMPLETED")) | .transcript_url' files.json)
curl --fail-with-body -sS "$url" | jq '{full_transcript, segments}'
curl --fail-with-body -sS "$jobs?limit=10" -H "$auth"
```

Each complete file has a temporary transcript URL. Fetch it without credentials and request the files page again if it expires. The transcript includes timestamps and speaker IDs when diarization is available. Job and file listings return `pagination.next_cursor`. To cancel an unfinished job, send `POST $jobs/$job_id/cancel` with an optional JSON body such as `{"reason":"Wrong recording"}`.

## LiteLLM Python SDK

```python
import litellm

jobs = litellm.gnani_batch()
job = jobs.create(
    model="gnani/prisma-v2.5",
    language_code="en-IN",
    urls=["https://your-media-host.example/meeting.flac"],
    with_diarization=True,
    num_speakers=2,
).json()
job_id = job["job_id"]
jobs.start(job_id)
status = jobs.get(job_id).json()
```

For local files, pass `files=[("meeting.wav", audio)]` with a binary file handle. For a webhook, pass `callback_url`. Use `options` for model-specific settings such as v3.0 key terms. To get results, call `jobs.files(job_id)` after completion. Both `files` and `list` accept `cursor`, `limit` and `status`. Use `jobs.cancel(job_id, reason="Wrong recording")` to cancel a job.

For async code, use `jobs = litellm.agnani_batch()` and await each job operation. Both clients read `GNANI_API_KEY` or accept an explicit `api_key`. They return HTTPX responses and raise `httpx.HTTPStatusError` on provider errors. Job states and per-file results retain their provider meanings, including `PARTIAL_FAILURE` for mixed outcomes. [Job states](https://docs.gnani.ai/api/STTBatch/Introduction#job-statuses)

For presigned uploads, call `jobs.create(..., upload_filenames=["meeting.flac"])` or post to `/gnani/stt/v3/batch/v2/jobs` with `source={"type":"direct_upload_url","files":[{"filename":"meeting.flac"}]}` alongside `config`. Upload each file to its returned URL with a credential-free PUT, then start the job. If the provider returns 404 for this route, submit a publicly accessible audio URL through the ordinary jobs endpoint. [Create job](https://docs.gnani.ai/api/STTBatch/Create_Job#presigned-upload-large-files)
