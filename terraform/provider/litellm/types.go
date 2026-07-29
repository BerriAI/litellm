package litellm

// ProviderConfig holds the configuration for the LiteLLM provider.
type ProviderConfig struct {
	APIBase            string
	APIKey             string
	InsecureSkipVerify bool
}

// ErrorResponse represents an error response from the API.
type ErrorResponse struct {
	Error struct {
		Message interface{} `json:"message"`
	} `json:"error"`
	Detail struct {
		Error string `json:"error"`
	} `json:"detail"`
}

// ModelResponse represents a response from the API containing model information.
type ModelResponse struct {
	ModelName     string                 `json:"model_name"`
	LiteLLMParams LiteLLMParams          `json:"litellm_params"`
	ModelInfo     ModelInfo              `json:"model_info"`
	Additional    map[string]interface{} `json:"additional"`
}

// ModelRequest represents a request to create or update a model.
type ModelRequest struct {
	ModelName     string                 `json:"model_name"`
	LiteLLMParams map[string]interface{} `json:"litellm_params"`
	ModelInfo     ModelInfo              `json:"model_info"`
	Additional    map[string]interface{} `json:"additional"`
}

// TeamResponse represents a response from the API containing team information.
type TeamResponse struct {
	TeamID                string                 `json:"team_id,omitempty"`
	TeamAlias             string                 `json:"team_alias,omitempty"`
	OrganizationID        string                 `json:"organization_id,omitempty"`
	Metadata              map[string]interface{} `json:"metadata,omitempty"`
	TPMLimit              *int                   `json:"tpm_limit,omitempty"`
	RPMLimit              *int                   `json:"rpm_limit,omitempty"`
	MaxBudget             *float64               `json:"max_budget,omitempty"`
	BudgetDuration        string                 `json:"budget_duration,omitempty"`
	Models                []string               `json:"models"`
	Blocked               bool                   `json:"blocked,omitempty"`
	TeamMemberPermissions []string               `json:"team_member_permissions,omitempty"`
}

// OrganizationResponse represents a response from the API containing organization information.
type OrganizationResponse struct {
	OrganizationID    string                 `json:"organization_id,omitempty"`
	OrganizationAlias string                 `json:"organization_alias,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	Models            []string               `json:"models,omitempty"`
	MaxBudget         *float64               `json:"max_budget,omitempty"`
	BudgetDuration    string                 `json:"budget_duration,omitempty"`
	TPMLimit          *int                   `json:"tpm_limit,omitempty"`
	RPMLimit          *int                   `json:"rpm_limit,omitempty"`
	Blocked           bool                   `json:"blocked,omitempty"`
}

// LiteLLMParams represents the parameters for LiteLLM.
type LiteLLMParams struct {
	CustomLLMProvider              string                 `json:"custom_llm_provider"`
	TPM                            int                    `json:"tpm,omitempty"`
	RPM                            int                    `json:"rpm,omitempty"`
	ReasoningEffort                string                 `json:"reasoning_effort,omitempty"`
	Thinking                       map[string]interface{} `json:"thinking,omitempty"`
	MergeReasoningContentInChoices bool                   `json:"merge_reasoning_content_in_choices,omitempty"`
	APIKey                         string                 `json:"api_key,omitempty"`
	APIBase                        string                 `json:"api_base,omitempty"`
	APIVersion                     string                 `json:"api_version,omitempty"`
	Model                          string                 `json:"model"`
	InputCostPerToken              float64                `json:"input_cost_per_token,omitempty"`
	OutputCostPerToken             float64                `json:"output_cost_per_token,omitempty"`
	InputCostPerPixel              float64                `json:"input_cost_per_pixel,omitempty"`
	OutputCostPerPixel             float64                `json:"output_cost_per_pixel,omitempty"`
	InputCostPerSecond             float64                `json:"input_cost_per_second,omitempty"`
	OutputCostPerSecond            float64                `json:"output_cost_per_second,omitempty"`
	AWSAccessKeyID                 string                 `json:"aws_access_key_id,omitempty"`
	AWSSecretAccessKey             string                 `json:"aws_secret_access_key,omitempty"`
	AWSRegionName                  string                 `json:"aws_region_name,omitempty"`
	AWSSessionName                 string                 `json:"aws_session_name,omitempty"`
	AWSRoleName                    string                 `json:"aws_role_name,omitempty"`
	VertexProject                  string                 `json:"vertex_project,omitempty"`
	VertexLocation                 string                 `json:"vertex_location,omitempty"`
	VertexCredentials              string                 `json:"vertex_credentials,omitempty"`
}

// ModelInfo represents information about a model.
type ModelInfo struct {
	ID        string `json:"id"`
	DBModel   bool   `json:"db_model"`
	BaseModel string `json:"base_model"`
	Tier      string `json:"tier"`
	Mode      string `json:"mode"`
	TeamID    string `json:"team_id,omitempty"`
}

// Key represents a LiteLLM API key.
type Key struct {
	Key                  string                 `json:"key,omitempty"`
	TokenID              string                 `json:"token_id,omitempty"`
	Models               []string               `json:"models"`
	Spend                float64                `json:"spend,omitempty"`
	MaxBudget            *float64               `json:"max_budget,omitempty"`
	UserID               string                 `json:"user_id,omitempty"`
	TeamID               string                 `json:"team_id,omitempty"`
	MaxParallelRequests  *int                   `json:"max_parallel_requests,omitempty"`
	Metadata             map[string]interface{} `json:"metadata,omitempty"`
	TPMLimit             *int                   `json:"tpm_limit,omitempty"`
	RPMLimit             *int                   `json:"rpm_limit,omitempty"`
	BudgetDuration       string                 `json:"budget_duration,omitempty"`
	AllowedCacheControls []string               `json:"allowed_cache_controls,omitempty"`
	SoftBudget           *float64               `json:"soft_budget,omitempty"`
	KeyAlias             string                 `json:"key_alias,omitempty"`
	Duration             string                 `json:"duration,omitempty"`
	Aliases              map[string]interface{} `json:"aliases,omitempty"`
	Config               map[string]interface{} `json:"config,omitempty"`
	Permissions          map[string]interface{} `json:"permissions,omitempty"`
	ModelMaxBudget       map[string]interface{} `json:"model_max_budget,omitempty"`
	ModelRPMLimit        map[string]interface{} `json:"model_rpm_limit,omitempty"`
	ModelTPMLimit        map[string]interface{} `json:"model_tpm_limit,omitempty"`
	Guardrails           []string               `json:"guardrails,omitempty"`
	Blocked              bool                   `json:"blocked"`
	Tags                 []string               `json:"tags,omitempty"`
}

// KeyResponse represents a response from the API containing key information.
type KeyResponse struct {
	Key string `json:"key"`
}

// MCPServerCostInfo represents cost information for MCP server tools.
type MCPServerCostInfo struct {
	DefaultCostPerQuery    float64            `json:"default_cost_per_query,omitempty"`
	ToolNameToCostPerQuery map[string]float64 `json:"tool_name_to_cost_per_query,omitempty"`
}

// MCPInfo represents MCP server information and configuration.
type MCPInfo struct {
	ServerName        string             `json:"server_name,omitempty"`
	Description       string             `json:"description,omitempty"`
	LogoURL           string             `json:"logo_url,omitempty"`
	MCPServerCostInfo *MCPServerCostInfo `json:"mcp_server_cost_info,omitempty"`
}

// MCPServerRequest represents a request to create or update an MCP server.
type MCPServerRequest struct {
	ServerID        string            `json:"server_id,omitempty"`
	ServerName      string            `json:"server_name"`
	Alias           string            `json:"alias,omitempty"`
	Description     string            `json:"description,omitempty"`
	Transport       string            `json:"transport"`
	SpecVersion     string            `json:"spec_version,omitempty"`
	AuthType        string            `json:"auth_type,omitempty"`
	URL             string            `json:"url"`
	MCPInfo         *MCPInfo          `json:"mcp_info,omitempty"`
	MCPAccessGroups []string          `json:"mcp_access_groups,omitempty"`
	Command         string            `json:"command,omitempty"`
	Args            []string          `json:"args,omitempty"`
	Env             map[string]string `json:"env,omitempty"`
}

// MCPServerResponse represents a response from the API containing MCP server information.
type MCPServerResponse struct {
	ServerID         string              `json:"server_id"`
	ServerName       string              `json:"server_name"`
	Alias            string              `json:"alias,omitempty"`
	Description      string              `json:"description,omitempty"`
	URL              string              `json:"url"`
	Transport        string              `json:"transport"`
	SpecVersion      string              `json:"spec_version,omitempty"`
	AuthType         string              `json:"auth_type,omitempty"`
	CreatedAt        string              `json:"created_at,omitempty"`
	CreatedBy        string              `json:"created_by,omitempty"`
	UpdatedAt        string              `json:"updated_at,omitempty"`
	UpdatedBy        string              `json:"updated_by,omitempty"`
	Teams            []map[string]string `json:"teams,omitempty"`
	MCPAccessGroups  []string            `json:"mcp_access_groups,omitempty"`
	MCPInfo          *MCPInfo            `json:"mcp_info,omitempty"`
	Status           string              `json:"status,omitempty"`
	LastHealthCheck  string              `json:"last_health_check,omitempty"`
	HealthCheckError string              `json:"health_check_error,omitempty"`
	Command          string              `json:"command,omitempty"`
	Args             []string            `json:"args,omitempty"`
	Env              map[string]string   `json:"env,omitempty"`
}

// CredentialRequest represents a request to create or update a credential.
type CredentialRequest struct {
	CredentialName   string                 `json:"credential_name"`
	CredentialInfo   map[string]interface{} `json:"credential_info,omitempty"`
	CredentialValues map[string]interface{} `json:"credential_values,omitempty"`
	ModelID          string                 `json:"model_id,omitempty"`
}

// CredentialResponse represents a response from the API containing credential information.
type CredentialResponse struct {
	CredentialName   string                 `json:"credential_name"`
	CredentialInfo   map[string]interface{} `json:"credential_info,omitempty"`
	CredentialValues map[string]interface{} `json:"credential_values,omitempty"`
}

// VectorStoreRequest represents a request to create or update a vector store.
type VectorStoreRequest struct {
	VectorStoreID          string                 `json:"vector_store_id,omitempty"`
	CustomLLMProvider      string                 `json:"custom_llm_provider"`
	VectorStoreName        string                 `json:"vector_store_name"`
	VectorStoreDescription string                 `json:"vector_store_description,omitempty"`
	VectorStoreMetadata    map[string]interface{} `json:"vector_store_metadata,omitempty"`
	LiteLLMCredentialName  string                 `json:"litellm_credential_name,omitempty"`
	LiteLLMParams          map[string]interface{} `json:"litellm_params,omitempty"`
}

// VectorStoreResponse represents a response from the API containing vector store information.
type VectorStoreResponse struct {
	VectorStoreID          string                 `json:"vector_store_id"`
	CustomLLMProvider      string                 `json:"custom_llm_provider"`
	VectorStoreName        string                 `json:"vector_store_name"`
	VectorStoreDescription string                 `json:"vector_store_description,omitempty"`
	VectorStoreMetadata    map[string]interface{} `json:"vector_store_metadata,omitempty"`
	CreatedAt              string                 `json:"created_at,omitempty"`
	UpdatedAt              string                 `json:"updated_at,omitempty"`
	LiteLLMCredentialName  string                 `json:"litellm_credential_name,omitempty"`
	LiteLLMParams          map[string]interface{} `json:"litellm_params,omitempty"`
}

// VectorStoreListResponse represents a response from the API containing a list of vector stores.
type VectorStoreListResponse struct {
	Object      string                `json:"object"`
	Data        []VectorStoreResponse `json:"data"`
	TotalCount  int                   `json:"total_count"`
	CurrentPage int                   `json:"current_page"`
	TotalPages  int                   `json:"total_pages"`
}

// VectorStoreDeleteRequest represents a request to delete a vector store.
type VectorStoreDeleteRequest struct {
	VectorStoreID string `json:"vector_store_id"`
}

// VectorStoreInfoRequest represents a request to get vector store information.
type VectorStoreInfoRequest struct {
	VectorStoreID string `json:"vector_store_id"`
}
