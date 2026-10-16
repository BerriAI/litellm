package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// mcpServerDetail intentionally omits env, credentials, and static_headers:
// those may hold secrets and must never reach data source state.
type mcpServerDetail struct {
	ServerID         string   `json:"server_id"`
	ServerName       string   `json:"server_name"`
	Alias            string   `json:"alias"`
	Description      string   `json:"description"`
	URL              string   `json:"url"`
	Transport        string   `json:"transport"`
	SpecVersion      string   `json:"spec_version"`
	AuthType         string   `json:"auth_type"`
	MCPAccessGroups  []string `json:"mcp_access_groups"`
	AllowedTools     []string `json:"allowed_tools"`
	ExtraHeaders     []string `json:"extra_headers"`
	Command          string   `json:"command"`
	Args             []string `json:"args"`
	AllowAllKeys     bool     `json:"allow_all_keys"`
	Status           string   `json:"status"`
	LastHealthCheck  string   `json:"last_health_check"`
	HealthCheckError string   `json:"health_check_error"`
	CreatedAt        string   `json:"created_at"`
	CreatedBy        string   `json:"created_by"`
	UpdatedAt        string   `json:"updated_at"`
	UpdatedBy        string   `json:"updated_by"`
}

func dataSourceLiteLLMMCPServer() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMMCPServerRead,

		Schema: map[string]*schema.Schema{
			"server_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Unique identifier of the MCP server to retrieve",
			},
			"server_name": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"alias": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"description": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"url": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"transport": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"spec_version": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"auth_type": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"mcp_access_groups": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"allowed_tools": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"extra_headers": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Names of request headers forwarded to the MCP server",
			},
			"command": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"args": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"allow_all_keys": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"status": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"last_health_check": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"health_check_error": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_by": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_by": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMMCPServerRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	serverID := d.Get("server_id").(string)

	endpoint := fmt.Sprintf("%s/%s", endpointMCPServerRead, serverID)
	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read MCP server: %w", err)
	}
	defer resp.Body.Close()

	var server mcpServerDetail
	if err := handleMCPAPIResponse(resp, &server, client); err != nil {
		if err.Error() == "mcp_server_not_found" {
			return fmt.Errorf("MCP server %q not found", serverID)
		}
		return fmt.Errorf("failed to read MCP server: %w", err)
	}

	d.SetId(GetStringValue(server.ServerID, serverID))
	d.Set("server_name", server.ServerName)
	d.Set("alias", server.Alias)
	d.Set("description", server.Description)
	d.Set("url", server.URL)
	d.Set("transport", server.Transport)
	d.Set("spec_version", server.SpecVersion)
	d.Set("auth_type", server.AuthType)
	d.Set("mcp_access_groups", server.MCPAccessGroups)
	d.Set("allowed_tools", server.AllowedTools)
	d.Set("extra_headers", server.ExtraHeaders)
	d.Set("command", server.Command)
	d.Set("args", server.Args)
	d.Set("allow_all_keys", server.AllowAllKeys)
	d.Set("status", server.Status)
	d.Set("last_health_check", server.LastHealthCheck)
	d.Set("health_check_error", server.HealthCheckError)
	d.Set("created_at", server.CreatedAt)
	d.Set("created_by", server.CreatedBy)
	d.Set("updated_at", server.UpdatedAt)
	d.Set("updated_by", server.UpdatedBy)

	log.Printf("[INFO] Successfully read MCP server with ID: %s", serverID)
	return nil
}

func dataSourceLiteLLMMCPServers() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMMCPServersRead,

		Schema: map[string]*schema.Schema{
			"team_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter to servers this team can access plus globally available servers",
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of the returned MCP servers",
			},
			"mcp_servers": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"server_id":      {Type: schema.TypeString, Computed: true},
						"server_name":    {Type: schema.TypeString, Computed: true},
						"alias":          {Type: schema.TypeString, Computed: true},
						"description":    {Type: schema.TypeString, Computed: true},
						"url":            {Type: schema.TypeString, Computed: true},
						"transport":      {Type: schema.TypeString, Computed: true},
						"spec_version":   {Type: schema.TypeString, Computed: true},
						"auth_type":      {Type: schema.TypeString, Computed: true},
						"allow_all_keys": {Type: schema.TypeBool, Computed: true},
						"status":         {Type: schema.TypeString, Computed: true},
						"created_at":     {Type: schema.TypeString, Computed: true},
						"updated_at":     {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMMCPServersRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	endpoint := endpointMCPServerRead
	if v, ok := d.GetOk("team_id"); ok {
		endpoint = fmt.Sprintf("%s?team_id=%s", endpointMCPServerRead, url.QueryEscape(v.(string)))
	}

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to list MCP servers: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing MCP servers"); err != nil {
		return err
	}

	var serverList []mcpServerDetail
	if err := json.NewDecoder(resp.Body).Decode(&serverList); err != nil {
		return fmt.Errorf("failed to decode MCP server list response: %w", err)
	}

	ids := make([]string, 0, len(serverList))
	servers := make([]map[string]interface{}, 0, len(serverList))
	for _, server := range serverList {
		ids = append(ids, server.ServerID)
		servers = append(servers, map[string]interface{}{
			"server_id":      server.ServerID,
			"server_name":    server.ServerName,
			"alias":          server.Alias,
			"description":    server.Description,
			"url":            server.URL,
			"transport":      server.Transport,
			"spec_version":   server.SpecVersion,
			"auth_type":      server.AuthType,
			"allow_all_keys": server.AllowAllKeys,
			"status":         server.Status,
			"created_at":     server.CreatedAt,
			"updated_at":     server.UpdatedAt,
		})
	}

	d.SetId(GetStringValue(d.Get("team_id").(string), "all"))
	d.Set("ids", ids)
	d.Set("mcp_servers", servers)

	log.Printf("[INFO] Successfully listed %d MCP servers", len(servers))
	return nil
}
