package litellm

import (
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/validation"
)

func resourceLiteLLMMCPServer() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMMCPServerCreate,
		Read:   resourceLiteLLMMCPServerRead,
		Update: resourceLiteLLMMCPServerUpdate,
		Delete: resourceLiteLLMMCPServerDelete,
		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"server_name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Name of the MCP server",
			},
			"alias": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "Alias for the MCP server",
			},
			"description": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Description of the MCP server",
			},
			"url": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "URL of the MCP server",
			},
			"transport": {
				Type:     schema.TypeString,
				Required: true,
				ValidateFunc: validation.StringInSlice([]string{
					"http",
					"sse",
					"stdio",
				}, false),
				Description: "Transport type for the MCP server (http, sse, stdio)",
			},
			"spec_version": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "2024-11-05",
				Description: "MCP specification version",
			},
			"auth_type": {
				Type:     schema.TypeString,
				Optional: true,
				Default:  "none",
				ValidateFunc: validation.StringInSlice([]string{
					"none",
					"api_key",
					"bearer",
					"bearer_token",
					"basic",
					"authorization",
					"oauth2",
					"aws_sigv4",
					"token",
					"oauth2_token_exchange",
					"oauth2_id_jag",
					"true_passthrough",
					"oauth_delegate",
				}, false),
				Description: "Authentication type the proxy uses against this server (none, api_key, bearer_token, basic, " +
					"authorization, oauth2, aws_sigv4, token, oauth2_token_exchange, oauth2_id_jag, true_passthrough, " +
					"oauth_delegate)",
			},
			"auth_value_wo": {
				Type:         schema.TypeString,
				Optional:     true,
				WriteOnly:    true,
				Sensitive:    true,
				RequiredWith: []string{"auth_value_wo_version"},
				Description: "Write-only credential for api_key, bearer_token, basic and authorization auth types " +
					"(the token or user:password). Sent to the proxy on every apply and never written to plan or " +
					"state. Requires Terraform 1.11 or later. Bump auth_value_wo_version to push a new value.",
			},
			"auth_value_wo_version": {
				Type:         schema.TypeInt,
				Optional:     true,
				RequiredWith: []string{"auth_value_wo"},
				Description:  "Version of auth_value_wo. Change it whenever auth_value_wo changes to trigger an update.",
			},
			"client_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "OAuth 2.0 client id the proxy presents to the server's token endpoint (oauth2 auth types)",
			},
			"client_secret_wo": {
				Type:         schema.TypeString,
				Optional:     true,
				WriteOnly:    true,
				Sensitive:    true,
				RequiredWith: []string{"client_secret_wo_version", "client_id"},
				Description: "Write-only OAuth 2.0 client secret paired with client_id. Sent to the proxy on every " +
					"apply and never written to plan or state. Requires Terraform 1.11 or later. Bump " +
					"client_secret_wo_version to push a new value.",
			},
			"client_secret_wo_version": {
				Type:         schema.TypeInt,
				Optional:     true,
				RequiredWith: []string{"client_secret_wo"},
				Description:  "Version of client_secret_wo. Change it whenever client_secret_wo changes to trigger an update.",
			},
			"mcp_access_groups": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of access groups for the MCP server",
			},
			"command": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Command to run for stdio transport",
			},
			"args": {
				Type:        schema.TypeList,
				Optional:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Arguments for the command (stdio transport)",
			},
			"env": {
				Type:        schema.TypeMap,
				Optional:    true,
				Sensitive:   true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Environment variables for the command (stdio transport)",
			},
			"mcp_info": {
				Type:        schema.TypeList,
				Optional:    true,
				MaxItems:    1,
				Description: "MCP server information and configuration",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"server_name": {
							Type:        schema.TypeString,
							Optional:    true,
							Description: "Server name in MCP info",
						},
						"description": {
							Type:        schema.TypeString,
							Optional:    true,
							Description: "Description in MCP info",
						},
						"logo_url": {
							Type:        schema.TypeString,
							Optional:    true,
							Description: "Logo URL for the MCP server",
						},
						"mcp_server_cost_info": {
							Type:        schema.TypeList,
							Optional:    true,
							MaxItems:    1,
							Description: "Cost information for MCP server tools",
							Elem: &schema.Resource{
								Schema: map[string]*schema.Schema{
									"default_cost_per_query": {
										Type:        schema.TypeFloat,
										Optional:    true,
										Description: "Default cost per query",
									},
									"tool_name_to_cost_per_query": {
										Type:        schema.TypeMap,
										Optional:    true,
										Elem:        &schema.Schema{Type: schema.TypeFloat},
										Description: "Map of tool names to their cost per query",
									},
								},
							},
						},
					},
				},
			},
			// Read-only computed fields
			"server_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Unique identifier for the MCP server",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the server was created",
			},
			"created_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User who created the server",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp when the server was last updated",
			},
			"updated_by": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "User who last updated the server",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Current status of the MCP server",
			},
			"last_health_check": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Timestamp of the last health check",
			},
			"health_check_error": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Error message from the last health check, if any",
			},
		},
	}
}
