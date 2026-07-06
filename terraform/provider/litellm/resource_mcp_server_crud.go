package litellm

import (
	"fmt"
	"log"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointMCPServerCreate = "/v1/mcp/server"
	endpointMCPServerUpdate = "/v1/mcp/server"
	endpointMCPServerRead   = "/v1/mcp/server"
	endpointMCPServerDelete = "/v1/mcp/server"
)

// Helper function to convert schema data to MCPServerRequest
func buildMCPServerRequest(d *schema.ResourceData) *MCPServerRequest {
	req := &MCPServerRequest{
		ServerName:  d.Get("server_name").(string),
		URL:         d.Get("url").(string),
		Transport:   d.Get("transport").(string),
		SpecVersion: d.Get("spec_version").(string),
		AuthType:    d.Get("auth_type").(string),
	}

	// Set optional fields
	if alias, ok := d.GetOk("alias"); ok {
		req.Alias = alias.(string)
	}
	if description, ok := d.GetOk("description"); ok {
		req.Description = description.(string)
	}
	if command, ok := d.GetOk("command"); ok {
		req.Command = command.(string)
	}

	// Handle access groups
	if accessGroups, ok := d.GetOk("mcp_access_groups"); ok {
		accessGroupsList := accessGroups.([]interface{})
		req.MCPAccessGroups = make([]string, len(accessGroupsList))
		for i, group := range accessGroupsList {
			req.MCPAccessGroups[i] = group.(string)
		}
	}

	// Handle args
	if args, ok := d.GetOk("args"); ok {
		argsList := args.([]interface{})
		req.Args = make([]string, len(argsList))
		for i, arg := range argsList {
			req.Args[i] = arg.(string)
		}
	}

	// Handle env
	if env, ok := d.GetOk("env"); ok {
		envMap := env.(map[string]interface{})
		req.Env = make(map[string]string)
		for k, v := range envMap {
			req.Env[k] = v.(string)
		}
	}

	// Handle mcp_info
	if mcpInfoList, ok := d.GetOk("mcp_info"); ok {
		mcpInfos := mcpInfoList.([]interface{})
		if len(mcpInfos) > 0 {
			mcpInfoMap := mcpInfos[0].(map[string]interface{})
			req.MCPInfo = &MCPInfo{}

			if serverName, ok := mcpInfoMap["server_name"]; ok {
				req.MCPInfo.ServerName = serverName.(string)
			}
			if description, ok := mcpInfoMap["description"]; ok {
				req.MCPInfo.Description = description.(string)
			}
			if logoURL, ok := mcpInfoMap["logo_url"]; ok {
				req.MCPInfo.LogoURL = logoURL.(string)
			}

			// Handle cost info
			if costInfoList, ok := mcpInfoMap["mcp_server_cost_info"]; ok {
				costInfos := costInfoList.([]interface{})
				if len(costInfos) > 0 {
					costInfoMap := costInfos[0].(map[string]interface{})
					req.MCPInfo.MCPServerCostInfo = &MCPServerCostInfo{}

					if defaultCost, ok := costInfoMap["default_cost_per_query"]; ok {
						req.MCPInfo.MCPServerCostInfo.DefaultCostPerQuery = defaultCost.(float64)
					}
					if toolCosts, ok := costInfoMap["tool_name_to_cost_per_query"]; ok {
						toolCostMap := toolCosts.(map[string]interface{})
						req.MCPInfo.MCPServerCostInfo.ToolNameToCostPerQuery = make(map[string]float64)
						for k, v := range toolCostMap {
							req.MCPInfo.MCPServerCostInfo.ToolNameToCostPerQuery[k] = v.(float64)
						}
					}
				}
			}
		}
	}

	return req
}

// Helper function to update schema data from MCPServerResponse
func updateSchemaFromResponse(d *schema.ResourceData, resp *MCPServerResponse) error {
	d.Set("server_id", resp.ServerID)
	d.Set("server_name", resp.ServerName)
	d.Set("alias", resp.Alias)
	d.Set("description", resp.Description)
	d.Set("url", resp.URL)
	d.Set("transport", resp.Transport)
	d.Set("spec_version", resp.SpecVersion)
	d.Set("auth_type", resp.AuthType)
	d.Set("created_at", resp.CreatedAt)
	d.Set("created_by", resp.CreatedBy)
	d.Set("updated_at", resp.UpdatedAt)
	d.Set("updated_by", resp.UpdatedBy)
	d.Set("status", resp.Status)
	d.Set("last_health_check", resp.LastHealthCheck)
	d.Set("health_check_error", resp.HealthCheckError)
	d.Set("command", resp.Command)

	// Set access groups
	if resp.MCPAccessGroups != nil {
		d.Set("mcp_access_groups", resp.MCPAccessGroups)
	}

	// Set args
	if resp.Args != nil {
		d.Set("args", resp.Args)
	}

	// Set env
	if resp.Env != nil {
		d.Set("env", resp.Env)
	}

	// Set mcp_info
	if resp.MCPInfo != nil {
		mcpInfoList := make([]map[string]interface{}, 1)
		mcpInfoMap := make(map[string]interface{})

		mcpInfoMap["server_name"] = resp.MCPInfo.ServerName
		mcpInfoMap["description"] = resp.MCPInfo.Description
		mcpInfoMap["logo_url"] = resp.MCPInfo.LogoURL

		if resp.MCPInfo.MCPServerCostInfo != nil {
			costInfoList := make([]map[string]interface{}, 1)
			costInfoMap := make(map[string]interface{})

			costInfoMap["default_cost_per_query"] = resp.MCPInfo.MCPServerCostInfo.DefaultCostPerQuery
			if resp.MCPInfo.MCPServerCostInfo.ToolNameToCostPerQuery != nil {
				costInfoMap["tool_name_to_cost_per_query"] = resp.MCPInfo.MCPServerCostInfo.ToolNameToCostPerQuery
			}

			costInfoList[0] = costInfoMap
			mcpInfoMap["mcp_server_cost_info"] = costInfoList
		}

		mcpInfoList[0] = mcpInfoMap
		d.Set("mcp_info", mcpInfoList)
	}

	return nil
}

func resourceLiteLLMMCPServerCreate(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	req := buildMCPServerRequest(d)

	resp, err := MakeRequest(client, "POST", endpointMCPServerCreate, req)
	if err != nil {
		return fmt.Errorf("failed to create MCP server: %w", err)
	}
	defer resp.Body.Close()

	var mcpResp MCPServerResponse
	if err := handleMCPAPIResponse(resp, &mcpResp, client); err != nil {
		return fmt.Errorf("failed to create MCP server: %w", err)
	}

	d.SetId(mcpResp.ServerID)

	// Update the state with the response data
	if err := updateSchemaFromResponse(d, &mcpResp); err != nil {
		return fmt.Errorf("failed to update state after create: %w", err)
	}

	log.Printf("[INFO] MCP server created with ID %s", mcpResp.ServerID)
	return nil
}

func resourceLiteLLMMCPServerRead(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	serverID := d.Id()
	endpoint := fmt.Sprintf("%s/%s", endpointMCPServerRead, serverID)

	resp, err := MakeRequest(client, "GET", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to read MCP server: %w", err)
	}
	defer resp.Body.Close()

	var mcpResp MCPServerResponse
	if err := handleMCPAPIResponse(resp, &mcpResp, client); err != nil {
		if err.Error() == "mcp_server_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read MCP server: %w", err)
	}

	// Update the state with the response data
	if err := updateSchemaFromResponse(d, &mcpResp); err != nil {
		return fmt.Errorf("failed to update state after read: %w", err)
	}

	return nil
}

func resourceLiteLLMMCPServerUpdate(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	req := buildMCPServerRequest(d)
	req.ServerID = d.Id() // Ensure we include the server ID for updates

	resp, err := MakeRequest(client, "PUT", endpointMCPServerUpdate, req)
	if err != nil {
		return fmt.Errorf("failed to update MCP server: %w", err)
	}
	defer resp.Body.Close()

	var mcpResp MCPServerResponse
	if err := handleMCPAPIResponse(resp, &mcpResp, client); err != nil {
		return fmt.Errorf("failed to update MCP server: %w", err)
	}

	// Update the state with the response data
	if err := updateSchemaFromResponse(d, &mcpResp); err != nil {
		return fmt.Errorf("failed to update state after update: %w", err)
	}

	log.Printf("[INFO] MCP server updated with ID %s", mcpResp.ServerID)
	return nil
}

func resourceLiteLLMMCPServerDelete(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	serverID := d.Id()
	endpoint := fmt.Sprintf("%s/%s", endpointMCPServerDelete, serverID)

	resp, err := MakeRequest(client, "DELETE", endpoint, nil)
	if err != nil {
		return fmt.Errorf("failed to delete MCP server: %w", err)
	}
	defer resp.Body.Close()

	// For delete operations, we expect a simple string response
	if resp.StatusCode != 200 {
		return fmt.Errorf("failed to delete MCP server: unexpected status code %d", resp.StatusCode)
	}

	d.SetId("")
	log.Printf("[INFO] MCP server deleted with ID %s", serverID)
	return nil
}

// retryMCPServerRead attempts to read an MCP server with exponential backoff
func retryMCPServerRead(d *schema.ResourceData, m interface{}, maxRetries int) error {
	var err error
	delay := 1 * time.Second
	maxDelay := 10 * time.Second

	for i := 0; i < maxRetries; i++ {
		log.Printf("[INFO] Attempting to read MCP server (attempt %d/%d)", i+1, maxRetries)

		err = resourceLiteLLMMCPServerRead(d, m)
		if err == nil {
			log.Printf("[INFO] Successfully read MCP server after %d attempts", i+1)
			return nil
		}

		// Check if this is a "server not found" error
		if err.Error() != "failed to read MCP server: mcp_server_not_found" {
			// If it's a different error, don't retry
			return err
		}

		if i < maxRetries-1 {
			log.Printf("[INFO] MCP server not found yet, retrying in %v...", delay)
			time.Sleep(delay)

			// Exponential backoff with a maximum delay
			delay *= 2
			if delay > maxDelay {
				delay = maxDelay
			}
		}
	}

	log.Printf("[WARN] Failed to read MCP server after %d attempts: %v", maxRetries, err)
	return err
}
