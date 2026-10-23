# rules

- Watsonx returns token strings and no numeric token IDs, so the normalized `token_ids` field stays null
- Request `parameters.return_tokens` explicitly and use the shared Watsonx auth service for IAM acquisition
- IAM URL overrides come from trusted deployment environment, never client request parameters

# references

- https://ibm.github.io/watsonx-ai-java-sdk/services/tokenization-service/
- https://www.ibm.com/docs/en/software-hub/5.1.x?topic=text-wx-ai-tokenize
- https://cloud.ibm.com/docs/apis/watsonx-ai
- Python credential and scope precedence in `litellm/llms/watsonx/common_utils.py`
