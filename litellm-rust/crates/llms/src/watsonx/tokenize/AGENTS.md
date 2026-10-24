# rules

- Watsonx returns token strings and no numeric token IDs, so the normalized `token_ids` field stays null
- Request `parameters.return_tokens` explicitly and use the shared Watsonx auth service for IAM acquisition

# references

- https://ibm.github.io/watsonx-ai-java-sdk/services/tokenization-service/
- https://www.ibm.com/docs/en/software-hub/5.1.x?topic=text-wx-ai-tokenize
- https://cloud.ibm.com/docs/apis/watsonx-ai
