# rules

- LiteLLM's OCR contract follows the Mistral OCR response shape. It is the normalized output every OCR provider returns, not a Mistral type
- Mistral, and Mistral models hosted on Azure AI and Vertex AI, speak it natively
- Cohere, Reducto, AWS Textract, Azure Document Intelligence and DeepSeek OCR on Vertex translate their native output into it in `llms/src/<provider>/ocr/`
- `OcrBoundingBox` is the corner coordinates providers copy into the normalized `bbox`
- Normalized `tables` and `keyValuePairs` stay open because providers pass through different native shapes
- `BaseOcrConfig`, OCR errors and inline-document helpers live in `llms/src/base_llm/ocr/`

# references

- https://docs.mistral.ai/api/endpoint/ocr
