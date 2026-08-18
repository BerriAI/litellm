package litellm

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
)

type Client struct {
	APIBase            string
	APIKey             string
	httpClient         *http.Client
	InsecureSkipVerify bool
}

func NewClient(apiBase, apiKey string, insecureSkipVerify bool) *Client {
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: insecureSkipVerify},
	}

	return &Client{
		APIBase:            apiBase,
		APIKey:             apiKey,
		httpClient:         &http.Client{Transport: tr},
		InsecureSkipVerify: insecureSkipVerify,
	}
}

// Organization member methods
func (c *Client) AddOrganizationMember(data map[string]interface{}) (map[string]interface{}, error) {
	return c.sendRequest("POST", "/organization/member_add", data)
}

func (c *Client) UpdateOrganizationMember(data map[string]interface{}) (map[string]interface{}, error) {
	return c.sendRequest("PATCH", "/organization/member_update", data)
}

func (c *Client) DeleteOrganizationMember(data map[string]interface{}) (map[string]interface{}, error) {
	return c.sendRequest("DELETE", "/organization/member_delete", data)
}

// Key-related methods
func (c *Client) CreateKey(key *Key) (*Key, error) {
	resp, err := c.sendRequest("POST", "/key/generate", key)
	if err != nil {
		return nil, err
	}

	return c.parseKeyResponse(resp)
}

func (c *Client) GetKey(keyID string) (*Key, error) {
	resp, err := c.sendRequest("GET", fmt.Sprintf("/key/info?key=%s", keyID), nil)
	if err != nil {
		return nil, err
	}

	return c.parseKeyResponse(resp)
}

func (c *Client) UpdateKey(key *Key) (*Key, error) {
	// Create a new map with only the fields that can be updated
	updateData := map[string]interface{}{
		"key":              key.Key,
		"team_id":          key.TeamID,
		"metadata":         key.Metadata,
		"budget_duration":  key.BudgetDuration,
		"key_alias":        key.KeyAlias,
		"aliases":          key.Aliases,
		"permissions":      key.Permissions,
		"model_max_budget": key.ModelMaxBudget,
		"model_rpm_limit":  key.ModelRPMLimit,
		"model_tpm_limit":  key.ModelTPMLimit,
		"blocked":          key.Blocked,
	}

	// Only add pointer fields if they are explicitly set
	if key.MaxBudget != nil {
		updateData["max_budget"] = *key.MaxBudget
	}
	if key.SoftBudget != nil {
		updateData["soft_budget"] = *key.SoftBudget
	}
	if key.MaxParallelRequests != nil {
		updateData["max_parallel_requests"] = *key.MaxParallelRequests
	}
	if key.TPMLimit != nil {
		updateData["tpm_limit"] = *key.TPMLimit
	}
	if key.RPMLimit != nil {
		updateData["rpm_limit"] = *key.RPMLimit
	}

	// Only add array fields if they are non-empty
	if len(key.Models) > 0 {
		updateData["models"] = key.Models
	}
	if len(key.Guardrails) > 0 {
		updateData["guardrails"] = key.Guardrails
	}
	if len(key.Tags) > 0 {
		updateData["tags"] = key.Tags
	}

	resp, err := c.sendRequest("POST", "/key/update", updateData)
	if err != nil {
		return nil, err
	}

	return c.parseKeyResponse(resp)
}

func (c *Client) DeleteKey(keyID string) error {
	payload := map[string]interface{}{
		"keys": []string{keyID},
	}
	_, err := c.sendRequest("POST", "/key/delete", payload)
	return err
}

func (c *Client) parseKeyResponse(resp map[string]interface{}) (*Key, error) {
	if resp == nil {
		return nil, fmt.Errorf("received nil response")
	}

	createdKey := &Key{}

	for k, v := range resp {
		if v == nil {
			continue
		}

		switch k {
		case "key":
			if s, ok := v.(string); ok {
				createdKey.Key = s
			}
		case "token_id":
			if s, ok := v.(string); ok {
				createdKey.TokenID = s
			}
		case "models":
			if models, ok := v.([]interface{}); ok {
				createdKey.Models = make([]string, len(models))
				for i, model := range models {
					if s, ok := model.(string); ok {
						createdKey.Models[i] = s
					}
				}
			}
		case "spend":
			if f, ok := v.(float64); ok {
				createdKey.Spend = f
			}
		case "max_budget":
			if f, ok := v.(float64); ok {
				createdKey.MaxBudget = &f
			}
		case "user_id":
			if s, ok := v.(string); ok {
				createdKey.UserID = s
			}
		case "team_id":
			if s, ok := v.(string); ok {
				createdKey.TeamID = s
			}
		case "max_parallel_requests":
			if i, ok := v.(float64); ok {
				val := int(i)
				createdKey.MaxParallelRequests = &val
			}
		case "metadata":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.Metadata = m
			}
		case "tpm_limit":
			if i, ok := v.(float64); ok {
				val := int(i)
				createdKey.TPMLimit = &val
			}
		case "rpm_limit":
			if i, ok := v.(float64); ok {
				val := int(i)
				createdKey.RPMLimit = &val
			}
		case "budget_duration":
			if s, ok := v.(string); ok {
				createdKey.BudgetDuration = s
			}
		case "soft_budget":
			if f, ok := v.(float64); ok {
				createdKey.SoftBudget = &f
			}
		case "key_alias":
			if s, ok := v.(string); ok {
				createdKey.KeyAlias = s
			}
		case "duration":
			if s, ok := v.(string); ok {
				createdKey.Duration = s
			}
		case "aliases":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.Aliases = m
			}
		case "config":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.Config = m
			}
		case "permissions":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.Permissions = m
			}
		case "model_max_budget":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.ModelMaxBudget = m
			}
		case "model_rpm_limit":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.ModelRPMLimit = m
			}
		case "model_tpm_limit":
			if m, ok := v.(map[string]interface{}); ok {
				createdKey.ModelTPMLimit = m
			}
		case "guardrails":
			if guardrails, ok := v.([]interface{}); ok {
				createdKey.Guardrails = make([]string, len(guardrails))
				for i, guardrail := range guardrails {
					if s, ok := guardrail.(string); ok {
						createdKey.Guardrails[i] = s
					}
				}
			}
		case "blocked":
			if b, ok := v.(bool); ok {
				createdKey.Blocked = b
			}
		case "tags":
			if tags, ok := v.([]interface{}); ok {
				createdKey.Tags = make([]string, len(tags))
				for i, tag := range tags {
					if s, ok := tag.(string); ok {
						createdKey.Tags[i] = s
					}
				}
			}
		}
	}

	return createdKey, nil
}

func (c *Client) sendRequest(method, path string, body interface{}) (map[string]interface{}, error) {
	url := c.APIBase + path

	var req *http.Request
	var err error

	if body != nil {
		jsonBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("error marshaling request body: %v", err)
		}
		log.Printf("Making %s request to %s with body:\n%s", method, url, c.redactSensitiveData(string(jsonBody)))
		req, err = http.NewRequest(method, url, bytes.NewBuffer(jsonBody))
	} else {
		log.Printf("Making %s request to %s", method, url)
		req, err = http.NewRequest(method, url, nil)
	}

	if err != nil {
		return nil, fmt.Errorf("error creating request: %v", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.APIKey)
	req.Header.Set("accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("error making request: %v", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %v", err)
	}

	log.Printf("Response status: %d", resp.StatusCode)
	log.Printf("Response body: %s", c.redactSensitiveData(string(bodyBytes)))

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API request failed with status code %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string]interface{}
	if err := json.Unmarshal(bodyBytes, &result); err != nil {
		if (method == "POST" || method == "PATCH" || method == "PUT" || method == "DELETE") &&
			(len(bodyBytes) == 0 || string(bodyBytes) == "null") {
			return make(map[string]interface{}), nil
		}
		return nil, fmt.Errorf("error parsing response JSON: %v\nResponse body: %s", err, string(bodyBytes))
	}

	return result, nil
}

var sensitiveLogFields = map[string]bool{
	"api_key":               true,
	"key":                   true,
	"token":                 true,
	"password":              true,
	"secret":                true,
	"credential":            true,
	"auth":                  true,
	"model_api_key":         true,
	"aws_access_key_id":     true,
	"aws_secret_access_key": true,
	"vertex_credentials":    true,
	"x-api-key":             true,
	"credential_values":     true,
}

func redactJSONValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		redacted := make(map[string]interface{}, len(typed))
		for k, v := range typed {
			if sensitiveLogFields[k] {
				redacted[k] = "[REDACTED]"
			} else {
				redacted[k] = redactJSONValue(v)
			}
		}
		return redacted
	case []interface{}:
		redacted := make([]interface{}, len(typed))
		for i, v := range typed {
			redacted[i] = redactJSONValue(v)
		}
		return redacted
	default:
		return value
	}
}

var sensitiveLogPatterns = []*regexp.Regexp{
	regexp.MustCompile(`"(api_key|key|token|password|secret|credential|auth)":\s*"[^"]*"`),
	regexp.MustCompile(`"(model_api_key|aws_access_key_id|aws_secret_access_key|vertex_credentials)":\s*"[^"]*"`),
	regexp.MustCompile(`"(x-api-key)":\s*"[^"]*"`),
}

func redactWithPatterns(data string) string {
	result := data
	for _, re := range sensitiveLogPatterns {
		result = re.ReplaceAllStringFunc(result, func(match string) string {
			parts := strings.SplitN(match, ":", 2)
			if len(parts) == 2 {
				return parts[0] + `: "[REDACTED]"`
			}
			return "[REDACTED]"
		})
	}
	return result
}

// redactSensitiveData masks sensitive information in logs
func (c *Client) redactSensitiveData(data string) string {
	var parsed interface{}
	if err := json.Unmarshal([]byte(data), &parsed); err != nil {
		return redactWithPatterns(data)
	}
	redactedBytes, err := json.Marshal(redactJSONValue(parsed))
	if err != nil {
		return redactWithPatterns(data)
	}
	return string(redactedBytes)
}
