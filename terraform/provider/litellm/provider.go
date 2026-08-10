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
		},
		DataSourcesMap: map[string]*schema.Resource{
			"litellm_credential":   dataSourceLiteLLMCredential(),
			"litellm_vector_store": dataSourceLiteLLMVectorStore(),
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
			"custom_headers": {
				Type:        schema.TypeMap,
				Optional:    true,
				Sensitive:   true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Optional HTTP headers to include on every request to the LiteLLM API (e.g. proxy or gateway headers)",
			},
		},
		ConfigureFunc: providerConfigure,
	}
}

// providerConfigure configures the provider with the given schema data.
func providerConfigure(d *schema.ResourceData) (interface{}, error) {
	return NewClient(ProviderConfig{
		APIBase:            d.Get("api_base").(string),
		APIKey:             d.Get("api_key").(string),
		InsecureSkipVerify: d.Get("insecure_skip_verify").(bool),
		CustomHeaders:      customHeadersFromSchema(d),
	}), nil
}

func customHeadersFromSchema(d *schema.ResourceData) map[string]string {
	v, ok := d.GetOk("custom_headers")
	if !ok {
		return map[string]string{}
	}
	raw := v.(map[string]interface{})
	headers := make(map[string]string, len(raw))
	for k, val := range raw {
		headers[k] = val.(string)
	}
	return headers
}
