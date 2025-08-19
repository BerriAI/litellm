# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Setup
```bash
# Install dependencies with Poetry
poetry install --with dev,proxy-dev

# Or install specific extras
poetry install --with dev          # Core development dependencies
poetry install --extras "proxy"    # Proxy server dependencies
poetry install --extras "extra_proxy"  # Additional proxy features (Prisma, Azure, GCS, Redis)
```

### Running the Application
```bash
# Start proxy server locally
uvicorn litellm.proxy.proxy_server:app --host localhost --port 4000 --reload

# Run with Docker dependencies
docker-compose up db prometheus  # Start dependent services
```

### Testing
```bash
# Run all tests
make test
# Or: poetry run pytest tests/

# Run unit tests only
make test-unit
# Or: poetry run pytest tests/litellm/

# Run integration tests
make test-integration
# Or: poetry run pytest tests/ -k "not litellm"

# Run specific test categories
poetry run pytest tests/llm_translation/  # LLM translation tests
poetry run pytest tests/router_unit_tests/  # Router tests
poetry run pytest tests/proxy_unit_tests/  # Proxy server tests
```

### Code Quality
```bash
# Run linting and type checking
make lint
# Or manually:
poetry run mypy litellm --ignore-missing-imports

# Format code with Black
poetry run black .

# Run Ruff for linting and formatting
poetry run ruff check .
poetry run ruff format .
```

## High-Level Architecture

### Core Components

1. **LiteLLM Core (`litellm/`)**: Main library providing unified LLM API interface
   - `main.py`: Core completion functions (completion, acompletion, streaming)
   - `utils.py`: Utility functions for API translation and model management
   - `router.py`: Load balancing and failover logic across multiple deployments
   - `cost_calculator.py`: Tracks and calculates costs across providers

2. **LLM Provider Integrations (`litellm/llms/`)**: Provider-specific implementations
   - Each provider has its own module handling API translation
   - Base classes define common interfaces
   - Supports 100+ LLM providers (OpenAI, Anthropic, Azure, Bedrock, Vertex AI, etc.)

3. **Proxy Server (`litellm/proxy/`)**: LLM Gateway for production deployments
   - `proxy_server.py`: FastAPI server providing OpenAI-compatible endpoints
   - Handles authentication, rate limiting, budgets, and usage tracking
   - Database-backed (Prisma) for key management and spend tracking
   - Supports virtual keys, team management, and model access control

4. **Router System (`litellm/router.py` & `router_strategy/`)**: Intelligent request routing
   - Load balancing strategies: least busy, lowest cost, lowest latency
   - Automatic failover and retry logic
   - Cooldown management for failed deployments
   - Tag-based and pattern-based routing

5. **Caching Layer (`litellm/caching/`)**: Multi-tier caching system
   - In-memory, Redis, and S3 cache implementations
   - Semantic caching with embeddings
   - Dual cache for combining different cache types
   - LLM response caching for cost reduction

6. **Observability (`litellm/integrations/`)**: Comprehensive logging and monitoring
   - 30+ integrations (Langfuse, DataDog, Prometheus, OpenTelemetry, etc.)
   - Custom callback system for extensibility
   - Standard logging payload format across all callbacks

7. **Moneta Integration (`moneta/`)**: Lago billing system integration
   - Pre-call entitlement checking
   - Post-call usage reporting
   - Thread-safe call metadata storage
   - Graceful error handling with fallback options

### Request Flow

1. **Client Request** → LiteLLM (via SDK or Proxy)
2. **Router Selection** → Chooses deployment based on strategy
3. **Pre-call Checks** → Budget limits, rate limits, entitlements (Moneta/Lago)
4. **Provider Translation** → Converts to provider-specific format
5. **API Call** → Sends to LLM provider
6. **Response Processing** → Standardizes response format
7. **Post-call Actions** → Usage tracking, cost calculation, callbacks
8. **Client Response** → Returns standardized OpenAI-format response

### Key Design Patterns

- **Provider Abstraction**: All providers implement common interfaces, allowing seamless switching
- **Async-First**: Core functions support both sync and async operations
- **Callback Architecture**: Extensible system for logging, monitoring, and custom actions
- **Retry/Fallback Logic**: Automatic handling of failures with configurable policies
- **Cost Tracking**: Built-in cost calculation and budget management
- **OpenAI Compatibility**: Maintains OpenAI API format for easy migration

### Database Schema (Prisma)

The proxy server uses PostgreSQL with Prisma ORM for:
- API key management
- Team and user management
- Usage and spend tracking
- Model configurations
- Budget limits and alerts

### Testing Strategy

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test end-to-end flows with real providers
- **Load Tests**: Performance testing for proxy server
- **Provider-Specific Tests**: Ensure correct translation for each provider

## Code Style Guidelines

- Follow Google Python Style Guide
- Use type hints for all functions
- Maintain async/sync parity for core functions
- Keep provider implementations isolated
- Use environment variables for configuration
- Implement proper error handling with custom exceptions