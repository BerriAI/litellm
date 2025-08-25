# Tech Stack

## Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Frontend Language | TypeScript | 5.x | Type-safe JavaScript for any UI components | Type safety and developer productivity |
| Frontend Framework | React (Admin UI) | 18.x | Admin dashboard and management UI | Component reusability and ecosystem |
| UI Component Library | Custom/Minimal | - | Basic admin components | Lightweight for API-first product |
| State Management | React Context | - | Simple state for admin UI | Minimal complexity for simple UI |
| Backend Language | Python | 3.8+ | Core library and proxy server | ML/AI ecosystem and library availability |
| Backend Framework | FastAPI | 0.104+ | High-performance async API server | Modern Python async support and OpenAPI |
| API Style | REST (OpenAI-compatible) | - | Primary API interface | Industry standard compatibility |
| Database | PostgreSQL | 13+ | Primary data store | ACID compliance and JSON support |
| Cache | Redis | 7.0+ | Session state and response caching | High-performance in-memory storage |
| File Storage | S3-compatible | - | File uploads and batch processing | Scalable object storage |
| Authentication | JWT + API Keys | - | Multi-method authentication | Flexible auth for different use cases |
| Frontend Testing | Jest/Vitest | Latest | Unit testing for any UI | Fast and reliable testing |
| Backend Testing | Pytest | 7.4+ | Comprehensive test suite | Python standard with async support |
| E2E Testing | Pytest | 7.4+ | End-to-end API testing | Unified testing framework |
| Build Tool | Poetry | 1.4+ | Dependency management and packaging | Modern Python packaging |
| Bundler | Webpack/Vite | Latest | Frontend bundling if needed | Modern build tooling |
| IaC Tool | Docker/Kubernetes | Latest | Container orchestration | Cloud-agnostic deployment |
| CI/CD | GitHub Actions | - | Automated testing and deployment | Native GitHub integration |
| Monitoring | Prometheus | 2.x | Metrics collection | Open-source and Kubernetes-native |
| Logging | Multiple (30+ integrations) | Various | Flexible logging backends | Customer choice of logging platform |
| CSS Framework | Tailwind CSS | 3.x | Utility-first CSS for admin UI | Rapid UI development |
