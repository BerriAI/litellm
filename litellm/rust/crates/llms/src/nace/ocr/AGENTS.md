# rules

- The model suffix sets `parse_mode` (`nace/parse-low`, `-medium`, `-high`), one cost map row per effort, so callers cannot pass `parse_mode`, `parser`, `source` or `corpus_sources`
- An https link whose file name has an extension Parse reads is sent as a `url` source. Every other document is uploaded through an upload grant, parsed as a `workspace_file` source and deleted when the job ends
- Guardrails see the source document before upload, as in `reducto/ocr`
- The API key goes only to the API base. The upload carries only the grant's token, and its URL must be the API base origin or https on a public address, whatever the download URL policy allows
- Follow-up calls (job polls, `content_url`, file deletion) go to the API base origin, never to a URL from elsewhere, and every poll must name the created job
- Cleanup runs when the call returns, with a 5 s timeout for each DELETE, never on a detached task. A cancelled call can leave its file and job, which is a documented limit. A successful job with `output.include_images` is kept, since its crop links die with it
- Error text is only the code and message of the Nace envelope, with `Bearer` values and `nace_sk_` keys masked, URL queries cut and the length capped. The exact API key is also masked in errors of the calls the adapter sends itself (upload grant, polls, content download, DELETEs) and in the job copy the response hooks see. The parse request error (`get_error_class`) and a failed job's error (`normalize_job`) mask by pattern only, because `get_error_class` and the sync response transform get no request headers. An upload error carries only its status
- `password` goes only as the top-level option, which the logging callbacks mask. Inside `extra_body` it is refused
- The top-level `include_image_base64` is never sent, since Nace returns images as links (`output.include_images`), not base64. `true` gets an unsupported-param 400 unless `drop_params` is on, and `false` or null is accepted
- Top-level `pages` takes zero-based non-negative integers. LiteLLM sorts and deduplicates them, sends consecutive runs as one-based inclusive `page_ranges`, and omits the `pages` key. A request with both top-level options set to non-null values gets a 400
- `output.include_page_markers` is forced on and `markdown` is always requested, so pages are split on the `--- Page N ---` lines. Blocks are not forced, so a caller gets only the formats it asked for
- With `page_ranges`, Nace keeps the document's page numbers, so the markers must follow the requested pages and each page's index is its page number minus one
- A result Nace truncates is fetched in full from `document.content_url`, and a native response gets that markdown in place of the preview and loses the link. A truncated workbook result is an error, never a silent preview
- A job answer without a `job_id` is an error. With `page_ranges`, only billed units count the pages

# references

- https://console.nace.ai/docs/perception/parse
- https://console.nace.ai/docs/perception/uploads
- https://console.nace.ai/docs/perception/jobs
- https://console.nace.ai/docs/perception/format-support
- https://console.nace.ai/docs/perception/pricing
