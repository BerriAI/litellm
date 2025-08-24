# Unified Project Structure

```plaintext
litellm/
├── .github/                    # CI/CD workflows
│   └── workflows/
│       ├── ci.yaml            # Test pipeline
│       ├── release.yaml       # Release automation
│       └── docker-publish.yaml
├── litellm/                    # Core library package
│   ├── llms/                   # Provider implementations
│   │   ├── openai.py
│   │   ├── anthropic.py
│   │   ├── azure.py
│   │   └── ...
│   ├── proxy/                  # Proxy server
│   │   ├── proxy_server.py    # FastAPI app
│   │   ├── auth/               # Authentication
│   │   ├── db/                 # Database layer
│   │   ├── management_endpoints/
│   │   └── pass_through_endpoints/
│   ├── router_strategy/        # Routing strategies
│   ├── caching/                # Cache implementations
│   ├── integrations/           # Observability integrations
│   ├── main.py                 # Core completion functions
│   ├── utils.py                # Utilities
│   ├── cost_calculator.py     # Cost tracking
│   └── budget_manager.py       # Budget enforcement
├── moneta/                     # Moneta/Lago integration
│   ├── __init__.py
│   ├── client.py               # Lago client
│   └── handlers.py             # Event handlers
├── tests/                      # Test suite
│   ├── litellm_tests/          # Unit tests
│   ├── proxy_unit_tests/       # Proxy tests
│   ├── router_unit_tests/      # Router tests
│   └── integration_tests/      # E2E tests
├── docs/                       # Documentation
│   ├── architecture.md         # This document
│   ├── getting-started.md
│   └── api-reference.md
├── deploy/                     # Deployment configs
│   ├── docker/
│   │   └── Dockerfile
│   ├── charts/                 # Helm charts
│   │   └── litellm-helm/
│   └── terraform/              # IaC configs
├── scripts/                    # Utility scripts
│   ├── setup.sh
│   └── test.sh
├── schema.prisma               # Database schema
├── docker-compose.yml          # Local development
├── pyproject.toml             # Poetry config
├── Makefile                   # Build commands
├── .env.example               # Environment template
└── README.md                  # Project readme
```
