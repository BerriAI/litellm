# LiteLLM Fullstack Architecture Document

## Introduction

This document outlines the complete fullstack architecture for LiteLLM, including backend systems, frontend implementation, and their integration. It serves as the single source of truth for AI-driven development, ensuring consistency across the entire technology stack.

This unified approach combines what would traditionally be separate backend and frontend architecture documents, streamlining the development process for modern fullstack applications where these concerns are increasingly intertwined.

### Starter Template or Existing Project

**Status:** Existing Production Project

LiteLLM is an established, production-ready LLM gateway that has been actively developed and deployed. This architecture document reflects the current implementation while documenting planned enhancements, particularly the Moneta/Lago billing integration currently being developed.

### Change Log

| Date | Version | Description | Author |
|------|---------|-------------|--------|
| 2025-01-24 | v1.0 | Initial comprehensive architecture documentation | Winston (Architect) |
| 2025-01-24 | v1.1 | Added Moneta integration details | Winston (Architect) |

## High Level Architecture

### Technical Summary

LiteLLM implements a unified LLM gateway architecture using FastAPI for the proxy server and Python for the core library, providing OpenAI-compatible APIs to 100+ LLM providers. The system features intelligent load balancing, automatic failover, comprehensive cost tracking, and multi-tenant management with PostgreSQL for persistence and Redis for caching. The architecture employs provider abstraction patterns, async-first design, and extensive observability integrations to achieve enterprise-grade reliability and scalability. Infrastructure is containerized with Docker and deployable to Kubernetes, with built-in support for serverless deployments, making it suitable for both cloud-native and on-premise installations.

### Platform and Infrastructure Choice

**Platform:** Cloud-agnostic with Docker/Kubernetes support
**Key Services:** PostgreSQL, Redis, Prometheus, Various LLM Provider APIs
**Deployment Host and Regions:** Configurable based on deployment needs - supports AWS, Azure, GCP, or on-premise

### Repository Structure

**Structure:** Monorepo
**Monorepo Tool:** Poetry workspaces
**Package Organization:** Core library + Proxy server + Integrations as separate packages

### High Level Architecture Diagram

```mermaid
graph TB
    subgraph "Client Applications"
        CA[Client Apps]
        SDK[LiteLLM SDK]
    end
    
    subgraph "LiteLLM Gateway"
        subgraph "API Layer"
            PROXY[FastAPI Proxy Server<br/>Port 4000]
            AUTH[Auth Middleware<br/>JWT/API Keys/SSO]
        end
        
        subgraph "Core Engine"
            ROUTER[Router<br/>Load Balancing]
            TRANS[Provider Translation Layer]
            COST[Cost Calculator]
            BUDGET[Budget Manager]
        end
        
        subgraph "Data Layer"
            CACHE[Cache Layer<br/>Redis/In-Memory/S3]
            DB[(PostgreSQL<br/>Prisma ORM)]
            QUEUE[Transaction Queue]
        end
    end
    
    subgraph "External Services"
        subgraph "LLM Providers"
            OPENAI[OpenAI]
            ANTHROPIC[Anthropic]
            AZURE[Azure OpenAI]
            VERTEX[Vertex AI]
            BEDROCK[AWS Bedrock]
            OTHER[100+ Providers]
        end
        
        subgraph "Integrations"
            MONETA[Moneta/Lago Billing]
            MONITOR[Monitoring<br/>Prometheus/DataDog]
            LOG[Logging<br/>Langfuse/Langsmith]
        end
    end
    
    CA --> SDK
    SDK --> PROXY
    CA --> PROXY
    PROXY --> AUTH
    AUTH --> ROUTER
    ROUTER --> TRANS
    TRANS --> COST
    COST --> BUDGET
    BUDGET --> MONETA
    
    ROUTER --> CACHE
    AUTH --> DB
    BUDGET --> DB
    COST --> DB
    DB --> QUEUE
    
    TRANS --> OPENAI
    TRANS --> ANTHROPIC
    TRANS --> AZURE
    TRANS --> VERTEX
    TRANS --> BEDROCK
    TRANS --> OTHER
    
    PROXY --> MONITOR
    PROXY --> LOG
```

### Architectural Patterns

- **API Gateway Pattern:** Single entry point for all LLM requests with centralized auth, rate limiting, and monitoring - _Rationale:_ Simplifies client integration and provides consistent interface across providers
- **Provider Abstraction Pattern:** Unified interface hiding provider-specific implementations - _Rationale:_ Enables seamless provider switching and multi-provider strategies
- **Load Balancer Pattern:** Intelligent request distribution across multiple deployments - _Rationale:_ Optimizes for cost, latency, or availability based on configuration
- **Circuit Breaker Pattern:** Automatic failure detection and recovery - _Rationale:_ Prevents cascade failures and improves system resilience
- **Repository Pattern:** Abstract data access logic with Prisma ORM - _Rationale:_ Enables testing and future database migration flexibility
- **Async-First Architecture:** Non-blocking I/O throughout the stack - _Rationale:_ Maximizes throughput for I/O-bound LLM operations
- **Multi-Tenant Architecture:** Organization/Team/User hierarchy with isolated resources - _Rationale:_ Supports enterprise deployments with complex access controls
- **Event-Driven Callbacks:** Extensible hook system for custom business logic - _Rationale:_ Allows customization without modifying core code
- **Cost-First Design:** Budget enforcement and tracking at every layer - _Rationale:_ Prevents unexpected LLM costs and enables chargeback models

## Tech Stack

### Technology Stack Table

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

## Data Models

### LiteLLM_VerificationToken
**Purpose:** API key management with permissions, budgets, and metadata

**Key Attributes:**
- token: string - Hashed API key
- key_name: string - Human-readable key identifier
- user_id: string - Associated user
- team_id: string - Associated team
- permissions: JSON - Granular permissions object
- spend: float - Current spend amount
- max_budget: float - Maximum allowed budget
- models: string[] - Allowed model access

#### TypeScript Interface
```typescript
interface VerificationToken {
  token: string;
  key_name?: string;
  key_alias?: string;
  user_id?: string;
  team_id?: string;
  permissions?: Record<string, any>;
  spend: number;
  max_budget?: number;
  models?: string[];
  metadata?: Record<string, any>;
  created_at: Date;
  expires?: Date;
}
```

#### Relationships
- Belongs to User (optional)
- Belongs to Team (optional)
- Has many SpendLogs

### LiteLLM_UserTable
**Purpose:** User account management with SSO integration

**Key Attributes:**
- user_id: string - Unique user identifier
- user_email: string - Email address
- user_role: string - System role (admin/user)
- teams: string[] - Team memberships
- max_budget: float - User-level budget
- spend: float - Current user spend

#### TypeScript Interface
```typescript
interface User {
  user_id: string;
  user_email?: string;
  user_role?: 'admin' | 'user' | 'viewer';
  teams?: string[];
  organization_id?: string;
  max_budget?: number;
  spend: number;
  user_metadata?: Record<string, any>;
  created_at: Date;
  updated_at: Date;
}
```

#### Relationships
- Has many VerificationTokens
- Belongs to many Teams
- Belongs to Organization

### LiteLLM_TeamTable
**Purpose:** Team-based access control and budget management

**Key Attributes:**
- team_id: string - Unique team identifier
- organization_id: string - Parent organization
- admins: string[] - Team admin user IDs
- members: string[] - Team member user IDs
- max_budget: float - Team-level budget
- spend: float - Current team spend
- models: string[] - Allowed models for team

#### TypeScript Interface
```typescript
interface Team {
  team_id: string;
  team_alias?: string;
  organization_id?: string;
  admins?: string[];
  members?: string[];
  max_budget?: number;
  spend: number;
  models?: string[];
  blocked?: boolean;
  metadata?: Record<string, any>;
  created_at: Date;
  updated_at: Date;
}
```

#### Relationships
- Has many Users (members)
- Has many VerificationTokens
- Belongs to Organization

### LiteLLM_SpendLogs
**Purpose:** Detailed request logging and cost tracking

**Key Attributes:**
- request_id: string - Unique request identifier
- api_key: string - Key used for request
- model: string - Model called
- total_tokens: integer - Token usage
- spend: float - Cost of request
- user: string - User identifier
- metadata: JSON - Request metadata
- payload: JSON - Request/response data

#### TypeScript Interface
```typescript
interface SpendLog {
  request_id: string;
  api_key?: string;
  model?: string;
  api_provider?: string;
  total_tokens?: number;
  prompt_tokens?: number;
  completion_tokens?: number;
  spend: number;
  user?: string;
  team_id?: string;
  metadata?: Record<string, any>;
  payload?: Record<string, any>;
  created_at: Date;
}
```

#### Relationships
- Belongs to VerificationToken
- Belongs to User (optional)
- Belongs to Team (optional)

## API Specification

### REST API Specification

```yaml
openapi: 3.0.0
info:
  title: LiteLLM Proxy API
  version: 1.0.0
  description: OpenAI-compatible LLM gateway with multi-provider support
servers:
  - url: http://localhost:4000
    description: Local development server
  - url: https://api.litellm.ai
    description: Production server

paths:
  /chat/completions:
    post:
      summary: Create chat completion
      operationId: createChatCompletion
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ChatCompletionRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatCompletionResponse'
  
  /embeddings:
    post:
      summary: Create embeddings
      operationId: createEmbedding
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EmbeddingRequest'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/EmbeddingResponse'
  
  /key/generate:
    post:
      summary: Generate API key
      operationId: generateKey
      security:
        - MasterKey: []
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                key_alias:
                  type: string
                team_id:
                  type: string
                models:
                  type: array
                  items:
                    type: string
                max_budget:
                  type: number
      responses:
        '200':
          description: Generated key
          content:
            application/json:
              schema:
                type: object
                properties:
                  key:
                    type: string
                  key_id:
                    type: string

components:
  schemas:
    ChatCompletionRequest:
      type: object
      required:
        - model
        - messages
      properties:
        model:
          type: string
        messages:
          type: array
          items:
            type: object
            properties:
              role:
                type: string
                enum: [system, user, assistant]
              content:
                type: string
        temperature:
          type: number
        max_tokens:
          type: integer
        stream:
          type: boolean
```

## Components

### LiteLLM Core Library
**Responsibility:** Core completion functions, provider abstraction, and model management

**Key Interfaces:**
- `completion()` - Synchronous completion API
- `acompletion()` - Asynchronous completion API
- `embedding()` - Text embedding generation
- `transcription()` - Audio transcription

**Dependencies:** httpx, openai, tiktoken, provider SDKs

**Technology Stack:** Python 3.8+, async/await, type hints

### Proxy Server
**Responsibility:** FastAPI server providing OpenAI-compatible endpoints with auth, rate limiting, and monitoring

**Key Interfaces:**
- OpenAI-compatible REST APIs
- Management endpoints for keys, teams, users
- Health and metrics endpoints
- WebSocket support for streaming

**Dependencies:** LiteLLM Core, FastAPI, Prisma, Redis

**Technology Stack:** FastAPI, Uvicorn, Gunicorn, Pydantic

### Router System
**Responsibility:** Intelligent load balancing and failover across multiple LLM deployments

**Key Interfaces:**
- `route()` - Select optimal deployment
- `mark_failure()` - Track deployment health
- `cooldown_deployment()` - Temporary failure handling

**Dependencies:** LiteLLM Core, Redis (optional)

**Technology Stack:** Python async, strategy pattern implementations

### Provider Translation Layer
**Responsibility:** Convert between OpenAI format and provider-specific formats

**Key Interfaces:**
- Provider-specific completion handlers
- Cost calculation per provider
- Token counting implementations
- Error standardization

**Dependencies:** Provider SDKs, tiktoken

**Technology Stack:** Python, provider-specific libraries

### Budget Manager
**Responsibility:** Real-time budget enforcement and spend tracking

**Key Interfaces:**
- `check_budget()` - Pre-call budget validation
- `update_spend()` - Post-call spend recording
- `get_usage()` - Usage reporting

**Dependencies:** PostgreSQL, Redis

**Technology Stack:** Prisma ORM, PostgreSQL, Redis

### Moneta Integration
**Responsibility:** Lago billing system integration with virtual currency conversion

**Key Interfaces:**
- `check_entitlement()` - Pre-call entitlement check
- `report_usage()` - Post-call usage reporting
- `convert_to_askii()` - USD to virtual currency conversion

**Dependencies:** Lago API, httpx

**Technology Stack:** Python, async HTTP client, thread-safe storage

### Component Diagrams

```mermaid
graph LR
    subgraph "Request Flow"
        REQ[API Request] --> MW[Middleware Stack]
        MW --> AUTH[Authentication]
        AUTH --> RL[Rate Limiter]
        RL --> BUD[Budget Check]
        BUD --> ROUT[Router]
        ROUT --> PROV[Provider Handler]
        PROV --> RESP[Response]
    end
    
    subgraph "Data Flow"
        PROV --> LOG[Logging]
        LOG --> SPEND[Spend Tracking]
        SPEND --> DB[(Database)]
        SPEND --> BILL[Billing System]
    end
```

## External APIs

### OpenAI API
- **Purpose:** Primary LLM provider for GPT models
- **Documentation:** https://platform.openai.com/docs
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** Bearer token (API key)
- **Rate Limits:** Varies by tier and model

**Key Endpoints Used:**
- `POST /chat/completions` - Chat completions
- `POST /embeddings` - Text embeddings
- `POST /audio/transcriptions` - Speech to text

**Integration Notes:** Native OpenAI format, no translation needed

### Anthropic API
- **Purpose:** Claude model provider
- **Documentation:** https://docs.anthropic.com
- **Base URL(s):** https://api.anthropic.com
- **Authentication:** X-API-Key header
- **Rate Limits:** Varies by tier

**Key Endpoints Used:**
- `POST /messages` - Claude chat completions
- `POST /complete` - Legacy completions

**Integration Notes:** Requires message format translation

### Azure OpenAI API
- **Purpose:** Enterprise Azure-hosted OpenAI models
- **Documentation:** https://docs.microsoft.com/azure/cognitive-services/openai
- **Base URL(s):** https://{resource}.openai.azure.com
- **Authentication:** api-key header or Azure AD
- **Rate Limits:** Configured per deployment

**Key Endpoints Used:**
- `POST /openai/deployments/{deployment}/chat/completions` - Chat completions
- `POST /openai/deployments/{deployment}/embeddings` - Embeddings

**Integration Notes:** Requires deployment name mapping

### Lago Billing API
- **Purpose:** Usage-based billing and metering
- **Documentation:** Internal/Moneta documentation
- **Base URL(s):** Configured per environment
- **Authentication:** Bearer token
- **Rate Limits:** Standard API limits

**Key Endpoints Used:**
- `POST /events` - Usage event reporting
- `GET /subscriptions/{id}/usage` - Usage retrieval
- `GET /customers/{id}/balance` - Balance checking

**Integration Notes:** Async event batching for performance

## Core Workflows

```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy Server
    participant A as Auth Module
    participant B as Budget Manager
    participant M as Moneta/Lago
    participant R as Router
    participant T as Translation Layer
    participant L as LLM Provider
    participant D as Database
    participant O as Observability

    C->>P: POST /chat/completions
    P->>A: Validate API Key
    A->>D: Check key permissions
    D-->>A: Key valid
    A->>B: Check budget
    B->>M: Check entitlement (optional)
    M-->>B: Entitlement OK
    B-->>P: Budget OK
    
    P->>R: Route request
    R->>R: Select deployment (strategy)
    R->>T: Transform request
    T->>L: Provider API call
    L-->>T: Provider response
    T->>T: Standardize response
    
    T->>B: Calculate cost
    B->>D: Update spend
    B->>M: Report usage
    T->>O: Log metrics
    T-->>P: OpenAI format response
    P-->>C: Response
    
    alt Error occurred
        L--xT: API error
        T->>R: Mark failure
        R->>R: Select fallback
        R->>T: Retry with fallback
    end
```

## Database Schema

```sql
-- Core authentication and authorization
CREATE TABLE "LiteLLM_VerificationToken" (
    "token" TEXT PRIMARY KEY,
    "key_name" TEXT,
    "key_alias" TEXT,
    "user_id" TEXT,
    "team_id" TEXT,
    "permissions" JSONB,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "max_budget" DOUBLE PRECISION,
    "models" TEXT[],
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "expires" TIMESTAMP,
    FOREIGN KEY ("user_id") REFERENCES "LiteLLM_UserTable"("user_id"),
    FOREIGN KEY ("team_id") REFERENCES "LiteLLM_TeamTable"("team_id")
);

-- User management
CREATE TABLE "LiteLLM_UserTable" (
    "user_id" TEXT PRIMARY KEY,
    "user_email" TEXT UNIQUE,
    "user_role" TEXT CHECK ("user_role" IN ('admin', 'user', 'viewer')),
    "teams" TEXT[],
    "organization_id" TEXT,
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "user_metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id")
);

-- Team management
CREATE TABLE "LiteLLM_TeamTable" (
    "team_id" TEXT PRIMARY KEY,
    "team_alias" TEXT,
    "organization_id" TEXT,
    "admins" TEXT[],
    "members" TEXT[],
    "max_budget" DOUBLE PRECISION,
    "spend" DOUBLE PRECISION DEFAULT 0,
    "models" TEXT[],
    "blocked" BOOLEAN DEFAULT FALSE,
    "metadata" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY ("organization_id") REFERENCES "LiteLLM_OrganizationTable"("organization_id")
);

-- Usage tracking
CREATE TABLE "LiteLLM_SpendLogs" (
    "request_id" TEXT PRIMARY KEY,
    "api_key" TEXT,
    "model" TEXT,
    "api_provider" TEXT,
    "total_tokens" INTEGER,
    "prompt_tokens" INTEGER,
    "completion_tokens" INTEGER,
    "spend" DOUBLE PRECISION,
    "user" TEXT,
    "team_id" TEXT,
    "metadata" JSONB DEFAULT '{}',
    "payload" JSONB,
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_api_key (api_key),
    INDEX idx_created_at (created_at),
    INDEX idx_user (user),
    INDEX idx_team_id (team_id)
);

-- Model configuration
CREATE TABLE "LiteLLM_ProxyModelTable" (
    "model_id" TEXT PRIMARY KEY,
    "model_name" TEXT NOT NULL,
    "litellm_params" JSONB NOT NULL,
    "model_info" JSONB DEFAULT '{}',
    "created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name)
);
```

## Frontend Architecture

### Component Architecture

#### Component Organization
```text
litellm/
├── ui/                         # Admin UI (if present)
│   ├── components/
│   │   ├── common/            # Shared components
│   │   ├── dashboard/         # Dashboard views
│   │   ├── keys/              # Key management
│   │   ├── teams/             # Team management
│   │   └── users/             # User management
│   ├── hooks/                 # React hooks
│   ├── services/              # API client
│   └── styles/                # CSS/Tailwind
```

#### Component Template
```typescript
// Minimal admin component example
interface KeyListProps {
  teamId?: string;
  userId?: string;
}

export const KeyList: React.FC<KeyListProps> = ({ teamId, userId }) => {
  const { keys, loading, error } = useKeys({ teamId, userId });
  
  if (loading) return <Spinner />;
  if (error) return <ErrorMessage error={error} />;
  
  return (
    <div className="key-list">
      {keys.map(key => (
        <KeyCard key={key.token} data={key} />
      ))}
    </div>
  );
};
```

### State Management Architecture

#### State Structure
```typescript
interface AppState {
  auth: {
    user: User | null;
    token: string | null;
  };
  keys: {
    items: VerificationToken[];
    loading: boolean;
    error: Error | null;
  };
  teams: {
    items: Team[];
    selected: string | null;
  };
}
```

#### State Management Patterns
- Use React Context for global state
- Local component state for UI-only concerns
- API response caching with React Query/SWR

### Routing Architecture

#### Route Organization
```text
/admin                  # Admin dashboard
/admin/keys            # API key management  
/admin/teams           # Team management
/admin/users           # User management
/admin/models          # Model configuration
/admin/usage           # Usage analytics
```

#### Protected Route Pattern
```typescript
const ProtectedRoute: React.FC<Props> = ({ children, requiredRole }) => {
  const { user } = useAuth();
  
  if (!user) return <Navigate to="/login" />;
  if (requiredRole && user.role !== requiredRole) {
    return <Navigate to="/unauthorized" />;
  }
  
  return <>{children}</>;
};
```

### Frontend Services Layer

#### API Client Setup
```typescript
class LiteLLMClient {
  private baseURL: string;
  private token: string;

  constructor(baseURL: string, token: string) {
    this.baseURL = baseURL;
    this.token = token;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseURL}${path}`, {
      ...options,
      headers: {
        'Authorization': `Bearer ${this.token}`,
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });
    
    if (!response.ok) {
      throw new APIError(response);
    }
    
    return response.json();
  }
  
  // API methods...
}
```

#### Service Example
```typescript
export const keyService = {
  async list(params?: { teamId?: string }): Promise<VerificationToken[]> {
    return client.request('/key/list', { 
      params 
    });
  },
  
  async generate(data: GenerateKeyRequest): Promise<KeyResponse> {
    return client.request('/key/generate', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  },
  
  async delete(keyId: string): Promise<void> {
    return client.request(`/key/delete`, {
      method: 'POST',
      body: JSON.stringify({ key_id: keyId }),
    });
  },
};
```

## Backend Architecture

### Service Architecture

#### Controller/Route Organization
```text
litellm/proxy/
├── proxy_server.py              # Main FastAPI app
├── management_endpoints/        # Admin endpoints
│   ├── key_management.py       # API key CRUD
│   ├── team_management.py      # Team CRUD
│   ├── user_management.py      # User CRUD
│   └── model_management.py     # Model config
├── pass_through_endpoints/     # Provider pass-through
│   ├── langfuse_passthrough.py
│   └── success_handler.py
└── auth/                        # Authentication
    ├── auth_checks.py
    ├── jwt_handler.py
    └── user_api_key_auth.py
```

#### Controller Template
```python
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional
from litellm.proxy.auth import user_api_key_auth

router = APIRouter(prefix="/key", tags=["key-management"])

@router.post("/generate")
async def generate_key(
    data: GenerateKeyRequest,
    user_api_key_dict: dict = Depends(user_api_key_auth)
):
    """Generate a new API key"""
    try:
        # Check permissions
        if not _is_admin(user_api_key_dict):
            raise HTTPException(status_code=403, detail="Admin access required")
        
        # Generate key
        key = await prisma_client.create_key(
            key_alias=data.key_alias,
            team_id=data.team_id,
            models=data.models,
            max_budget=data.max_budget
        )
        
        return {"key": key.token, "key_id": key.key_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Database Architecture

#### Schema Design
```sql
-- See Database Schema section above for complete DDL
-- Additional indexes for performance
CREATE INDEX idx_spend_logs_date_model ON "LiteLLM_SpendLogs" (created_at, model);
CREATE INDEX idx_verification_token_team ON "LiteLLM_VerificationToken" (team_id);
CREATE INDEX idx_user_table_org ON "LiteLLM_UserTable" (organization_id);
```

#### Data Access Layer
```python
class PrismaClient:
    """Database access layer using Prisma ORM"""
    
    async def create_key(
        self,
        key_alias: str,
        team_id: Optional[str] = None,
        models: Optional[List[str]] = None,
        max_budget: Optional[float] = None
    ) -> VerificationToken:
        """Create a new API key"""
        token = generate_token()
        hashed_token = hash_token(token)
        
        key = await self.db.litellm_verificationtoken.create({
            "data": {
                "token": hashed_token,
                "key_alias": key_alias,
                "team_id": team_id,
                "models": models or [],
                "max_budget": max_budget,
                "created_at": datetime.utcnow()
            }
        })
        
        return key
    
    async def get_key_spend(self, token: str) -> float:
        """Get current spend for a key"""
        result = await self.db.litellm_spendlogs.aggregate(
            where={"api_key": token},
            _sum={"spend": True}
        )
        return result._sum.spend or 0.0
```

### Authentication and Authorization

#### Auth Flow
```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy
    participant A as Auth Module
    participant D as Database
    participant J as JWT Handler

    C->>P: Request with API Key/JWT
    P->>A: Validate credentials
    
    alt API Key Auth
        A->>D: Lookup hashed key
        D-->>A: Key details + permissions
    else JWT Auth
        A->>J: Verify JWT
        J-->>A: User claims
        A->>D: Lookup user permissions
        D-->>A: User details + permissions
    end
    
    A->>A: Check route permissions
    A-->>P: Auth context
    P->>P: Process request with context
```

#### Middleware/Guards
```python
async def user_api_key_auth(
    request: Request,
    api_key: str = Depends(get_api_key)
) -> dict:
    """Authentication middleware for API endpoints"""
    
    # Check cache first
    cached_key = await redis_client.get(f"key:{api_key}")
    if cached_key:
        return json.loads(cached_key)
    
    # Database lookup
    hashed_key = hash_token(api_key)
    key_data = await prisma_client.get_key(hashed_key)
    
    if not key_data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Check expiration
    if key_data.expires and key_data.expires < datetime.utcnow():
        raise HTTPException(status_code=401, detail="API key expired")
    
    # Cache for performance
    await redis_client.setex(
        f"key:{api_key}",
        300,  # 5 minute TTL
        json.dumps(key_data.dict())
    )
    
    return key_data
```

## Unified Project Structure

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

## Development Workflow

### Local Development Setup

#### Prerequisites
```bash
# Python 3.8 or higher
python --version

# Poetry for dependency management
curl -sSL https://install.python-poetry.org | python3 -

# Docker for services
docker --version

# PostgreSQL client (optional)
psql --version
```

#### Initial Setup
```bash
# Clone repository
git clone https://github.com/BerriAI/litellm.git
cd litellm

# Install dependencies
poetry install --with dev,proxy-dev

# Copy environment template
cp .env.example .env

# Start Docker services
docker-compose up -d db redis prometheus

# Run database migrations
poetry run prisma migrate dev

# Seed initial data (optional)
poetry run python scripts/seed.py
```

#### Development Commands
```bash
# Start all services
docker-compose up -d
poetry run uvicorn litellm.proxy.proxy_server:app --reload --port 4000

# Start frontend only (if admin UI present)
cd ui && npm run dev

# Start backend only
poetry run uvicorn litellm.proxy.proxy_server:app --reload

# Run tests
poetry run pytest tests/
make test-unit
make test-integration
```

### Environment Configuration

#### Required Environment Variables
```bash
# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:4000
NEXT_PUBLIC_ENVIRONMENT=development

# Backend (.env)
DATABASE_URL=postgresql://user:pass@localhost:5432/litellm
REDIS_URL=redis://localhost:6379
LITELLM_MASTER_KEY=sk-1234567890
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Shared
NODE_ENV=development
LOG_LEVEL=debug
```

## Deployment Architecture

### Deployment Strategy

**Frontend Deployment:**
- **Platform:** Static hosting (Vercel/Netlify) or containerized
- **Build Command:** `cd ui && npm run build`
- **Output Directory:** `ui/dist`
- **CDN/Edge:** CloudFlare or cloud provider CDN

**Backend Deployment:**
- **Platform:** Kubernetes, Docker Swarm, or cloud PaaS
- **Build Command:** `docker build -f deploy/docker/Dockerfile .`
- **Deployment Method:** Rolling update with health checks

### CI/CD Pipeline
```yaml
name: Deploy
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install --with dev
      - name: Run tests
        run: poetry run pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t litellm:${{ github.sha }} .
      - name: Push to registry
        run: |
          docker tag litellm:${{ github.sha }} ${{ secrets.REGISTRY }}/litellm:latest
          docker push ${{ secrets.REGISTRY }}/litellm:latest
  
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/litellm-proxy \
            litellm=${{ secrets.REGISTRY }}/litellm:latest \
            --namespace=production
```

### Environments

| Environment | Frontend URL | Backend URL | Purpose |
|------------|--------------|-------------|---------|
| Development | http://localhost:3000 | http://localhost:4000 | Local development |
| Staging | https://staging.litellm.ai | https://api-staging.litellm.ai | Pre-production testing |
| Production | https://app.litellm.ai | https://api.litellm.ai | Live environment |

## Security and Performance

### Security Requirements

**Frontend Security:**
- CSP Headers: `default-src 'self'; script-src 'self' 'unsafe-inline';`
- XSS Prevention: React's automatic escaping + input sanitization
- Secure Storage: API keys never stored in localStorage, use httpOnly cookies

**Backend Security:**
- Input Validation: Pydantic models for all inputs
- Rate Limiting: 100 req/min per key (configurable)
- CORS Policy: Explicit allowed origins only

**Authentication Security:**
- Token Storage: Hashed in database (SHA256)
- Session Management: JWT with 24h expiry, refresh tokens
- Password Policy: Not applicable (API key based)

### Performance Optimization

**Frontend Performance:**
- Bundle Size Target: < 200KB initial load
- Loading Strategy: Code splitting, lazy loading
- Caching Strategy: Browser cache + service worker

**Backend Performance:**
- Response Time Target: p99 < 100ms overhead
- Database Optimization: Connection pooling, query optimization
- Caching Strategy: Redis for hot data, in-memory LRU cache

## Testing Strategy

### Testing Pyramid
```text
        E2E Tests
        /        \
    Integration Tests
    /            \
Frontend Unit  Backend Unit
```

### Test Organization

#### Frontend Tests
```text
ui/tests/
├── unit/              # Component tests
├── integration/       # API integration
└── e2e/              # User flows
```

#### Backend Tests
```text
tests/
├── litellm_tests/     # Core library units
├── proxy_unit_tests/  # Proxy server units
├── router_unit_tests/ # Router logic
└── integration_tests/ # Provider integration
```

#### E2E Tests
```text
tests/e2e/
├── auth_flows.py
├── api_workflows.py
└── admin_operations.py
```

### Test Examples

#### Frontend Component Test
```typescript
describe('KeyList', () => {
  it('displays API keys', async () => {
    const keys = [
      { token: 'sk-123', key_alias: 'Test Key' }
    ];
    
    render(<KeyList keys={keys} />);
    
    expect(screen.getByText('Test Key')).toBeInTheDocument();
  });
});
```

#### Backend API Test
```python
@pytest.mark.asyncio
async def test_generate_key():
    """Test API key generation"""
    response = await client.post(
        "/key/generate",
        json={"key_alias": "Test Key"},
        headers={"Authorization": f"Bearer {MASTER_KEY}"}
    )
    
    assert response.status_code == 200
    assert "key" in response.json()
    assert response.json()["key"].startswith("sk-")
```

#### E2E Test
```python
@pytest.mark.asyncio
async def test_complete_flow():
    """Test complete request flow"""
    # Generate key
    key_response = await generate_test_key()
    api_key = key_response["key"]
    
    # Make completion request
    completion = await client.post(
        "/chat/completions",
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
    
    assert completion.status_code == 200
    assert "choices" in completion.json()
```

## Coding Standards

### Critical Fullstack Rules

- **Type Safety:** All functions must have type hints, use Pydantic for validation
- **API Consistency:** All endpoints return OpenAI-compatible format
- **Error Handling:** Use standard error handler, never expose internal errors
- **Async First:** Use async/await for all I/O operations
- **Cost Tracking:** Every LLM call must record cost and usage
- **Configuration:** Access config only through settings object, never env directly
- **Logging:** Use structured logging with request_id correlation
- **Testing:** Minimum 80% code coverage for new features

### Naming Conventions

| Element | Frontend | Backend | Example |
|---------|----------|---------|---------|
| Components | PascalCase | - | `KeyList.tsx` |
| Hooks | camelCase with 'use' | - | `useAuth.ts` |
| API Routes | - | kebab-case | `/api/key-management` |
| Database Tables | - | PascalCase with prefix | `LiteLLM_UserTable` |

## Error Handling Strategy

### Error Flow
```mermaid
sequenceDiagram
    participant C as Client
    participant P as Proxy
    participant E as Error Handler
    participant L as Logger
    participant M as Monitoring

    C->>P: Request
    P->>P: Process
    P--xP: Error occurs
    P->>E: Handle error
    E->>L: Log error details
    E->>M: Send metrics
    E->>E: Classify error
    E->>E: Sanitize message
    E-->>P: Error response
    P-->>C: Standardized error
```

### Error Response Format
```typescript
interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
    timestamp: string;
    requestId: string;
  };
}
```

### Frontend Error Handling
```typescript
class APIError extends Error {
  code: string;
  details?: Record<string, any>;
  
  constructor(response: Response, body: ApiError) {
    super(body.error.message);
    this.code = body.error.code;
    this.details = body.error.details;
  }
}

// Global error boundary
const ErrorBoundary: React.FC = ({ children }) => {
  return (
    <ErrorBoundaryComponent
      fallback={<ErrorFallback />}
      onError={(error) => {
        console.error(error);
        // Send to monitoring
      }}
    >
      {children}
    </ErrorBoundaryComponent>
  );
};
```

### Backend Error Handling
```python
from fastapi import HTTPException
from litellm.proxy.utils import StandardErrorResponse

class ProxyException(HTTPException):
    """Standard proxy exception"""
    
    def __init__(
        self,
        message: str,
        code: str,
        status_code: int = 500,
        details: dict = None
    ):
        self.code = code
        self.details = details
        super().__init__(
            status_code=status_code,
            detail=StandardErrorResponse(
                code=code,
                message=message,
                details=details,
                request_id=get_request_id()
            ).dict()
        )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    if isinstance(exc, ProxyException):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail
        )
    
    # Don't expose internal errors
    return JSONResponse(
        status_code=500,
        content=StandardErrorResponse(
            code="internal_error",
            message="An internal error occurred",
            request_id=get_request_id()
        ).dict()
    )
```

## Monitoring and Observability

### Monitoring Stack

- **Frontend Monitoring:** Browser console + optional Sentry
- **Backend Monitoring:** Prometheus + Grafana
- **Error Tracking:** Sentry or Rollbar (optional)
- **Performance Monitoring:** Prometheus metrics + custom dashboards

### Key Metrics

**Frontend Metrics:**
- Core Web Vitals
- JavaScript errors
- API response times
- User interactions

**Backend Metrics:**
- Request rate
- Error rate
- Response time
- Database query performance

## Checklist Results Report

*Ready to validate architecture completeness. Would you like me to output the full architecture document now, or shall we proceed with running the architect checklist?*