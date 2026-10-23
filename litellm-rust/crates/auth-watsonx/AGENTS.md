# rules

- Acquire IAM bearer tokens through the injected HTTP client and share cached tokens across calls
- Never include credential request bodies or token response bodies in errors
- IAM endpoint overrides come from trusted configuration, never an inference request body

# references

- https://cloud.ibm.com/docs/apis/iam-identity-token-api
