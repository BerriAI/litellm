# rules

- `transformation.rs` sends the Mistral OCR request to Azure AI's `/providers/mistral/azure/ocr` path
- `cohere_parse_transformation.rs` reuses `cohere/ocr` against Azure AI's `/providers/cohere/v2/parse` path
- `document_intelligence/` translates Azure Document Intelligence's async analyze operation into the shared OCR format

# references

- https://learn.microsoft.com/en-us/rest/api/aiservices/document-models/analyze-document
