# Preview acceptance fixture

This is a standalone simulated app for validating LiteReview's browser, recording and hosted preview pipeline. It is not part of LiteLLM and makes no external provider calls. Do not describe it as production behavior or provider E2E evidence.

From the repository root:

```sh
python -m http.server 3100 --directory ui/litereview-preview-fixture
```

Open http://localhost:3100, start recording, click **Successful retry** and **Provider failure**, then stop/save the recording. Include the saved recording in the preview and label it as a validation fixture. No dependencies or credentials are required. The draft PR must not be merged.
