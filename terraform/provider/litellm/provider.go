package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// Provider returns a terraform.ResourceProvider.
func Provider() *schema.Provider {
	return &schema.Provider{
		ResourcesMap: map[string]*schema.Resource{
			"litellm_model":                   resourceLiteLLMModel(),
			"litellm_team":                    ResourceLiteLLMTeam(),
			"litellm_organization":            resourceLiteLLMOrganization(),
			"litellm_organization_member":     resourceLiteLLMOrganizationMember(),
			"litellm_organization_member_add": resourceLiteLLMOrganizationMemberAdd(),
			"litellm_team_member":             resourceLiteLLMTeamMember(),
			"litellm_team_member_add":         resourceLiteLLMTeamMemberAdd(),
			"litellm_key":                     resourceKey(),
			"litellm_mcp_server":              resourceLiteLLMMCPServer(),
			"litellm_credential":              resourceLiteLLMCredential(),
			"litellm_vector_store":            resourceLiteLLMVectorStore(),
			"litellm_fallback":                resourceLiteLLMFallback(),
			"litellm_key_block":               resourceLiteLLMKeyBlock(),
			"litellm_team_block":              resourceLiteLLMTeamBlock(),
			"litellm_access_group":            resourceLiteLLMAccessGroup(),
			"litellm_unified_access_group":    resourceLiteLLMUnifiedAccessGroup(),
			"litellm_guardrail":               resourceLiteLLMGuardrail(),
			"litellm_prompt":                  resourceLiteLLMPrompt(),
			"litellm_agent":                   resourceLiteLLMAgent(),
			"litellm_search_tool":             resourceLiteLLMSearchTool(),
			"litellm_user":                    resourceLiteLLMUser(),
			"litellm_budget":                  resourceLiteLLMBudget(),
			"litellm_tag":                     resourceLiteLLMTag(),
			"litellm_project":                 resourceLiteLLMProject(),
		},
		DataSourcesMap: map[string]*schema.Resource{
			"litellm_credential":            dataSourceLiteLLMCredential(),
			"litellm_vector_store":          dataSourceLiteLLMVectorStore(),
			"litellm_fallback":              dataSourceLiteLLMFallback(),
			"litellm_access_group":          dataSourceLiteLLMAccessGroup(),
			"litellm_access_groups":         dataSourceLiteLLMAccessGroups(),
			"litellm_unified_access_group":  dataSourceLiteLLMUnifiedAccessGroup(),
			"litellm_unified_access_groups": dataSourceLiteLLMUnifiedAccessGroups(),
			"litellm_guardrail":             dataSourceLiteLLMGuardrail(),
			"litellm_guardrails":            dataSourceLiteLLMGuardrails(),
			"litellm_prompt":                dataSourceLiteLLMPrompt(),
			"litellm_prompts":               dataSourceLiteLLMPrompts(),
			"litellm_agent":                 dataSourceLiteLLMAgent(),
			"litellm_agents":                dataSourceLiteLLMAgents(),
			"litellm_search_tool":           dataSourceLiteLLMSearchTool(),
			"litellm_search_tools":          dataSourceLiteLLMSearchTools(),
			"litellm_user":                  dataSourceLiteLLMUser(),
			"litellm_users":                 dataSourceLiteLLMUsers(),
			"litellm_budget":                dataSourceLiteLLMBudget(),
			"litellm_budgets":               dataSourceLiteLLMBudgets(),
			"litellm_tag":                   dataSourceLiteLLMTag(),
			"litellm_tags":                  dataSourceLiteLLMTags(),
			"litellm_project":               dataSourceLiteLLMProject(),
			"litellm_projects":              dataSourceLiteLLMProjects(),
			"litellm_key":                   dataSourceLiteLLMKey(),
			"litellm_keys":                  dataSourceLiteLLMKeys(),
			"litellm_team":                  dataSourceLiteLLMTeam(),
			"litellm_teams":                 dataSourceLiteLLMTeams(),
			"litellm_model":                 dataSourceLiteLLMModel(),
			"litellm_models":                dataSourceLiteLLMModels(),
			"litellm_organization":          dataSourceLiteLLMOrganization(),
			"litellm_organizations":         dataSourceLiteLLMOrganizations(),
			"litellm_mcp_server":            dataSourceLiteLLMMCPServer(),
			"litellm_mcp_servers":           dataSourceLiteLLMMCPServers(),
		},
		Schema: map[string]*schema.Schema{
			"api_base": {
				Type:        schema.TypeString,
				Required:    true,
				Sensitive:   false,
				DefaultFunc: schema.EnvDefaultFunc("LITELLM_API_BASE", nil),
				Description: "The base URL of the LiteLLM API",
			},
			"api_key": {
				Type:        schema.TypeString,
				Required:    true,
				Sensitive:   true,
				DefaultFunc: schema.EnvDefaultFunc("LITELLM_API_KEY", nil),
				Description: "The API key for authenticating with LiteLLM",
			},
			"insecure_skip_verify": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     false,
				DefaultFunc: schema.EnvDefaultFunc("LITELLM_INSECURE_SKIP_VERIFY", false),
				Description: "Skip TLS certificate verification. Only use for development or when using self-signed certificates",
			},
		},
		ConfigureFunc: providerConfigure,
	}
}

// providerConfigure configures the provider with the given schema data.
func providerConfigure(d *schema.ResourceData) (interface{}, error) {
	config := ProviderConfig{
		APIBase:            d.Get("api_base").(string),
		APIKey:             d.Get("api_key").(string),
		InsecureSkipVerify: d.Get("insecure_skip_verify").(bool),
	}

	return NewClient(config.APIBase, config.APIKey, config.InsecureSkipVerify), nil
}
