package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
	"time"

	"github.com/hashicorp/go-cty/cty"
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

		Issuer:                d.Get("issuer").(string),
		AuthorizationURL:      d.Get("authorization_url").(string),
		TokenURL:              d.Get("token_url").(string),
		RegistrationURL:       d.Get("registration_url").(string),
		OAuth2Flow:            d.Get("oauth2_flow").(string),
		TokenExchangeEndpoint: d.Get("token_exchange_endpoint").(string),
		Audience:              d.Get("audience").(string),
		SubjectTokenType:      d.Get("subject_token_type").(string),
		TokenExchangeProfile:  d.Get("token_exchange_profile").(string),
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

	req.Credentials = mcpCredentialsFromResource(d)

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

// mcpCredentialsFromResource builds the credentials object from the write-only
// secrets and their non-secret companions; nil when none is configured so the
// proxy keeps whatever it already stores.
func mcpCredentialsFromResource(d *schema.ResourceData) *MCPCredentials {
	creds := &MCPCredentials{
		TokenEndpointAuthMethod:    d.Get("token_endpoint_auth_method").(string),
		UpstreamResource:           d.Get("upstream_resource").(string),
		ClientPrivateKeyID:         d.Get("client_private_key_id").(string),
		ClientAssertionSigningAlg:  d.Get("client_assertion_signing_alg").(string),
		IDJAGResourceTokenEndpoint: d.Get("id_jag_resource_token_endpoint").(string),
		IDJAGResource:              d.Get("id_jag_resource").(string),
		IDJAGResourceClientID:      d.Get("id_jag_resource_client_id").(string),
		IDJAGSubjectSource:         d.Get("id_jag_subject_source").(string),
		ClientPrivateKey:           writeOnlyString(d, "client_private_key_wo"),
		IDJAGResourceClientSecret:  writeOnlyString(d, "id_jag_resource_client_secret_wo"),
		AuthValue:                  writeOnlyString(d, "auth_value_wo"),
		ClientID:                   d.Get("client_id").(string),
		ClientSecret:               writeOnlyString(d, "client_secret_wo"),
	}
	for _, v := range d.Get("scopes").([]interface{}) {
		creds.Scopes = append(creds.Scopes, v.(string))
	}
	if reflect.DeepEqual(*creds, MCPCredentials{}) {
		return nil
	}
	return creds
}

// Helper function to update schema data from MCPServerResponse
func updateSchemaFromResponse(d *schema.ResourceData, resp *MCPServerResponse) error {
	d.Set("issuer", resp.Issuer)
	d.Set("authorization_url", resp.AuthorizationURL)
	d.Set("token_url", resp.TokenURL)
	d.Set("registration_url", resp.RegistrationURL)
	d.Set("oauth2_flow", resp.OAuth2Flow)
	d.Set("token_exchange_endpoint", resp.TokenExchangeEndpoint)
	d.Set("audience", resp.Audience)
	d.Set("subject_token_type", resp.SubjectTokenType)
	d.Set("token_exchange_profile", resp.TokenExchangeProfile)
	var credentials struct {
		Scopes           []string `json:"scopes"`
		UpstreamResource string   `json:"upstream_resource"`
	}
	// Some proxy versions return credentials as an opaque string.
	_ = json.Unmarshal(resp.Credentials, &credentials)
	d.Set("scopes", credentials.Scopes)
	d.Set("upstream_resource", credentials.UpstreamResource)
	d.Set("server_id", resp.ServerID)
	d.Set("server_name", resp.ServerName)
	d.Set("alias", resp.Alias)
	d.Set("description", resp.Description)
	d.Set("url", resp.URL)
	d.Set("transport", resp.Transport)
	if resp.SpecVersion != "" {
		d.Set("spec_version", resp.SpecVersion)
	}
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

	if !mcpInfoHasModeledFields(resp.MCPInfo) {
		d.Set("mcp_info", []interface{}{})
	} else {
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

func mcpInfoHasModeledFields(info *MCPInfo) bool {
	return info != nil &&
		(info.ServerName != "" ||
			info.Description != "" ||
			info.LogoURL != "" ||
			info.MCPServerCostInfo != nil)
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

	encoded, err := json.Marshal(req)
	if err != nil {
		return err
	}
	var body map[string]interface{}
	if err := json.Unmarshal(encoded, &body); err != nil {
		return err
	}
	authChanged := d.HasChange("auth_type")
	upstreamChanged := d.HasChange("url") || d.HasChange("issuer")
	oauthConfigured := false
	switch req.AuthType {
	case "oauth2", "oauth2_token_exchange", "oauth2_id_jag", "true_passthrough", "oauth_delegate":
		oauthConfigured = true
	}
	resendOAuthSettings := oauthConfigured && (upstreamChanged || authChanged)
	if resendOAuthSettings {
		body["preserve_explicit_oauth_settings"] = true
	}
	setMCPStringUpdates(d, body, resendOAuthSettings, "issuer", "authorization_url", "token_url", "registration_url",
		"oauth2_flow", "token_exchange_endpoint", "audience", "subject_token_type", "token_exchange_profile")

	credentials, ok := body["credentials"].(map[string]interface{})
	if !ok {
		credentials = make(map[string]interface{})
	}
	resendOAuthCredentials := authChanged && oauthConfigured
	setMCPStringUpdates(d, credentials, resendOAuthSettings, "token_endpoint_auth_method")
	setMCPStringUpdates(d, credentials, resendOAuthCredentials, "upstream_resource",
		"client_private_key_id", "client_assertion_signing_alg", "id_jag_resource_token_endpoint",
		"id_jag_resource", "id_jag_resource_client_id", "id_jag_subject_source")
	delete(credentials, "scopes")
	if d.HasChange("scopes") || (resendOAuthCredentials && mcpFieldConfigured(d, "scopes")) {
		credentials["scopes"] = d.Get("scopes").([]interface{})
	}

	for _, field := range []string{"client_private_key", "id_jag_resource_client_secret"} {
		if d.HasChange(field + "_wo_version") {
			if _, ok := credentials[field]; !ok {
				credentials[field] = nil
			}
		} else if !resendOAuthCredentials {
			delete(credentials, field)
		}
	}
	// An issuer or URL change can invalidate a client. Resend credentials only on an
	// explicit client edit so a later apply cannot silently rebind that client.
	if req.AuthType == "oauth2" && !authChanged {
		secretChanged := d.HasChange("client_secret_wo_version")
		clientChanged := d.HasChange("client_id")
		if !secretChanged && (!clientChanged || upstreamChanged) {
			delete(credentials, "client_secret")
		}
		if !secretChanged && !clientChanged && !d.HasChange("client_private_key_wo_version") {
			delete(credentials, "client_id")
		}
	}
	if len(credentials) == 0 {
		delete(body, "credentials")
	} else {
		body["credentials"] = credentials
	}

	resp, err := MakeRequest(client, "PUT", endpointMCPServerUpdate, body)
	if err != nil {
		return fmt.Errorf("failed to update MCP server: %w", err)
	}
	defer resp.Body.Close()

	var mcpResp MCPServerResponse
	if err := handleMCPAPIResponse(resp, &mcpResp, client); err != nil {
		return fmt.Errorf("failed to update MCP server: %w", err)
	}

	if err := updateSchemaFromResponse(d, &mcpResp); err != nil {
		return fmt.Errorf("failed to update state after update: %w", err)
	}

	log.Printf("[INFO] MCP server updated with ID %s", mcpResp.ServerID)
	return nil
}

func mcpFieldConfigured(d *schema.ResourceData, field string) bool {
	raw, diags := d.GetRawConfigAt(cty.GetAttrPath(field))
	return !diags.HasError() && raw.IsKnown() && !raw.IsNull()
}

func setMCPStringUpdates(d *schema.ResourceData, body map[string]interface{}, resendConfigured bool, fields ...string) {
	for _, field := range fields {
		delete(body, field)
		if !d.HasChange(field) && !(resendConfigured && mcpFieldConfigured(d, field)) {
			continue
		}
		if value := d.Get(field).(string); value != "" {
			body[field] = value
		} else {
			body[field] = nil
		}
	}
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
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
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
