# Data Models

## LiteLLM_VerificationToken
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

### TypeScript Interface
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

### Relationships
- Belongs to User (optional)
- Belongs to Team (optional)
- Has many SpendLogs

## LiteLLM_UserTable
**Purpose:** User account management with SSO integration

**Key Attributes:**
- user_id: string - Unique user identifier
- user_email: string - Email address
- user_role: string - System role (admin/user)
- teams: string[] - Team memberships
- max_budget: float - User-level budget
- spend: float - Current user spend

### TypeScript Interface
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

### Relationships
- Has many VerificationTokens
- Belongs to many Teams
- Belongs to Organization

## LiteLLM_TeamTable
**Purpose:** Team-based access control and budget management

**Key Attributes:**
- team_id: string - Unique team identifier
- organization_id: string - Parent organization
- admins: string[] - Team admin user IDs
- members: string[] - Team member user IDs
- max_budget: float - Team-level budget
- spend: float - Current team spend
- models: string[] - Allowed models for team

### TypeScript Interface
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

### Relationships
- Has many Users (members)
- Has many VerificationTokens
- Belongs to Organization

## LiteLLM_SpendLogs
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

### TypeScript Interface
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

### Relationships
- Belongs to VerificationToken
- Belongs to User (optional)
- Belongs to Team (optional)
