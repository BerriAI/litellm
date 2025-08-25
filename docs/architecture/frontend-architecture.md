# Frontend Architecture

## Component Architecture

### Component Organization
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

### Component Template
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

## State Management Architecture

### State Structure
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

### State Management Patterns
- Use React Context for global state
- Local component state for UI-only concerns
- API response caching with React Query/SWR

## Routing Architecture

### Route Organization
```text
/admin                  # Admin dashboard
/admin/keys            # API key management  
/admin/teams           # Team management
/admin/users           # User management
/admin/models          # Model configuration
/admin/usage           # Usage analytics
```

### Protected Route Pattern
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

## Frontend Services Layer

### API Client Setup
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

### Service Example
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
