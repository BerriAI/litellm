# Directory Structure

When adding a new provider, you need to create a directory for the provider that follows the following structure:

```
litellm/llms/
└── provider_name/
    ├── completion/
    │   ├── handler.py
    │   └── transformation.py
    ├── chat/
    │   ├── handler.py
    │   └── transformation.py
    ├── embed/
    │   ├── handler.py
    │   └── transformation.py
    └── rerank/
        ├── handler.py
        └── transformation.py
```

