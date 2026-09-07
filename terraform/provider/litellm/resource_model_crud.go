package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// retryModelRead attempts to read a model with exponential backoff.
// It handles the case where resourceLiteLLMModelRead returns nil but clears the ID
// (eventual consistency: model created but not yet visible on read-back).
func retryModelRead(d *schema.ResourceData, m interface{}, maxRetries int) error {
	delay := 1 * time.Second
	maxDelay := 10 * time.Second
	modelID := d.Id()

	for i := 0; i < maxRetries; i++ {
		log.Printf("[INFO] Attempting to read model (attempt %d/%d)", i+1, maxRetries)

		err := resourceLiteLLMModelRead(d, m)
		if err == nil {
			if d.Id() != "" {
				log.Printf("[INFO] Successfully read model after %d attempts", i+1)
				return nil
			}
			// Read returned nil but cleared the ID — model not yet visible (eventual consistency).
			// Restore the ID so we can retry.
			d.SetId(modelID)
			log.Printf("[INFO] Model not found yet (eventual consistency), retrying in %v...", delay)
		} else {
			log.Printf("[INFO] Read error, retrying in %v: %v", delay, err)
		}

		if i < maxRetries-1 {
			time.Sleep(delay)
			delay *= 2
			if delay > maxDelay {
				delay = maxDelay
			}
		}
	}

	log.Printf("[WARN] Failed to read model after %d attempts", maxRetries)
	return fmt.Errorf("model %s not found after %d read attempts post-create; the model may have been created successfully — re-running apply should resolve this", modelID, maxRetries)
}

const (
	endpointModelNew    = "/model/new"
	endpointModelUpdate = "/model/%s/update"
	endpointModelInfo   = "/v2/model/info"
	endpointModelDelete = "/model/delete"
)

func modelFieldConfigured(d *schema.ResourceData, name string) bool {
	if _, configured := d.GetOkExists(name); configured {
		return true
	}
	if d.HasChange(name) {
		return true
	}
	raw := d.GetRawConfig()
	if !raw.IsKnown() || raw.IsNull() || !raw.Type().HasAttribute(name) {
		return false
	}
	return !raw.GetAttr(name).IsNull()
}

func configuredModelInt(d *schema.ResourceData, name string) *int {
	if !modelFieldConfigured(d, name) {
		return nil
	}
	value := d.Get(name).(int)
	return &value
}

func configuredModelBool(d *schema.ResourceData, name string) *bool {
	if !modelFieldConfigured(d, name) {
		return nil
	}
	value := d.Get(name).(bool)
	return &value
}

func configuredModelFloat(d *schema.ResourceData, name string) *float64 {
	if !modelFieldConfigured(d, name) {
		return nil
	}
	value := d.Get(name).(float64)
	return &value
}

func configuredModelString(d *schema.ResourceData, name string) *string {
	if !modelFieldConfigured(d, name) {
		return nil
	}
	value := d.Get(name).(string)
	return &value
}

func configuredModelStringList(d *schema.ResourceData, name string) *[]string {
	if !modelFieldConfigured(d, name) {
		return nil
	}
	value := expandStringList(d.Get(name).([]interface{}))
	return &value
}

func modelRoutingBase(customLLMProvider, model string) string {
	prefix := customLLMProvider + "/"
	if customLLMProvider != "" && strings.HasPrefix(model, prefix) {
		return strings.TrimPrefix(model, prefix)
	}
	return model
}

func createOrUpdateModel(d *schema.ResourceData, m interface{}, isUpdate bool) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	// Construct the model name in the format "custom_llm_provider/base_model"
	customLLMProvider := d.Get("custom_llm_provider").(string)
	baseModel := d.Get("base_model").(string)
	modelName := fmt.Sprintf("%s/%s", customLLMProvider, baseModel)

	// Pricing base_model, decoupled from routing. When pricing_base_model is
	// set it feeds model_info.base_model (the cost-lookup key) WITHOUT changing
	// the routing string above; otherwise base_model drives pricing as before.
	pricingBaseModel := baseModel
	if v, ok := d.GetOk("pricing_base_model"); ok && v.(string) != "" {
		pricingBaseModel = v.(string)
	}

	// Generate a UUID for new models
	modelID := d.Id()
	if !isUpdate {
		modelID = uuid.New().String()
	}

	// Create thinking configuration if enabled
	var thinking map[string]interface{}
	if d.Get("thinking_enabled").(bool) {
		thinking = map[string]interface{}{
			"type":          "enabled",
			"budget_tokens": d.Get("thinking_budget_tokens").(int),
		}
	}

	// Build the base litellm_params as a map to allow for additional parameters
	litellmParams := map[string]interface{}{
		"custom_llm_provider":                customLLMProvider,
		"model":                              modelName,
		"merge_reasoning_content_in_choices": d.Get("merge_reasoning_content_in_choices").(bool),
	}

	// Add optional parameters only if they have values
	if tpm := d.Get("tpm").(int); tpm > 0 {
		litellmParams["tpm"] = tpm
	}
	if rpm := d.Get("rpm").(int); rpm > 0 {
		litellmParams["rpm"] = rpm
	}
	// Only include cost fields if explicitly set (non-zero)
	if inputCostPerMillion := d.Get("input_cost_per_million_tokens").(float64); inputCostPerMillion > 0 {
		litellmParams["input_cost_per_token"] = inputCostPerMillion / 1000000.0
	}
	if outputCostPerMillion := d.Get("output_cost_per_million_tokens").(float64); outputCostPerMillion > 0 {
		litellmParams["output_cost_per_token"] = outputCostPerMillion / 1000000.0
	}
	if apiKey := d.Get("model_api_key").(string); apiKey != "" {
		litellmParams["api_key"] = apiKey
	}
	if apiBase := d.Get("model_api_base").(string); apiBase != "" {
		litellmParams["api_base"] = apiBase
	}
	if apiVersion := d.Get("api_version").(string); apiVersion != "" {
		litellmParams["api_version"] = apiVersion
	}
	if inputCostPerPixel := d.Get("input_cost_per_pixel").(float64); inputCostPerPixel > 0 {
		litellmParams["input_cost_per_pixel"] = inputCostPerPixel
	}
	if outputCostPerPixel := d.Get("output_cost_per_pixel").(float64); outputCostPerPixel > 0 {
		litellmParams["output_cost_per_pixel"] = outputCostPerPixel
	}
	if inputCostPerSecond := d.Get("input_cost_per_second").(float64); inputCostPerSecond > 0 {
		litellmParams["input_cost_per_second"] = inputCostPerSecond
	}
	if outputCostPerSecond := d.Get("output_cost_per_second").(float64); outputCostPerSecond > 0 {
		litellmParams["output_cost_per_second"] = outputCostPerSecond
	}
	if awsAccessKeyID := d.Get("aws_access_key_id").(string); awsAccessKeyID != "" {
		litellmParams["aws_access_key_id"] = awsAccessKeyID
	}
	if awsSecretAccessKey := d.Get("aws_secret_access_key").(string); awsSecretAccessKey != "" {
		litellmParams["aws_secret_access_key"] = awsSecretAccessKey
	}
	if awsRegionName := d.Get("aws_region_name").(string); awsRegionName != "" {
		litellmParams["aws_region_name"] = awsRegionName
	}
	if awsSessionName := d.Get("aws_session_name").(string); awsSessionName != "" {
		litellmParams["aws_session_name"] = awsSessionName
	}
	if awsRoleName := d.Get("aws_role_name").(string); awsRoleName != "" {
		litellmParams["aws_role_name"] = awsRoleName
	}
	if vertexProject := d.Get("vertex_project").(string); vertexProject != "" {
		litellmParams["vertex_project"] = vertexProject
	}
	if vertexLocation := d.Get("vertex_location").(string); vertexLocation != "" {
		litellmParams["vertex_location"] = vertexLocation
	}
	if vertexCredentials := d.Get("vertex_credentials").(string); vertexCredentials != "" {
		litellmParams["vertex_credentials"] = vertexCredentials
	}
	if reasoningEffort := d.Get("reasoning_effort").(string); reasoningEffort != "" {
		litellmParams["reasoning_effort"] = reasoningEffort
	}
	if thinking != nil {
		litellmParams["thinking"] = thinking
	}

	// Add additional parameters if provided
	if additionalParams, ok := d.GetOk("additional_litellm_params"); ok {
		var dropParams []string

		for key, value := range additionalParams.(map[string]interface{}) {
			// Convert string values to appropriate types where possible
			if strValue, ok := value.(string); ok {
				// Check if it's JSON (starts with [ or {)
				trimmedValue := strings.TrimSpace(strValue)
				if strings.HasPrefix(trimmedValue, "[") || strings.HasPrefix(trimmedValue, "{") {
					var parsedValue interface{}
					if err := json.Unmarshal([]byte(strValue), &parsedValue); err == nil {
						// Successfully parsed JSON
						if key == "additional_drop_params" {
							// Handle drop params specially
							if dropList, ok := parsedValue.([]interface{}); ok {
								for _, item := range dropList {
									if paramStr, ok := item.(string); ok {
										dropParams = append(dropParams, paramStr)
									}
								}
							}
							continue // Don't add to litellmParams
						} else {
							litellmParams[key] = parsedValue
						}
					} else {
						// Not valid JSON, apply existing conversion logic
						if strValue == "true" {
							litellmParams[key] = true
						} else if strValue == "false" {
							litellmParams[key] = false
						} else {
							// Try to convert numeric strings
							if intValue, err := strconv.Atoi(strValue); err == nil {
								litellmParams[key] = intValue
							} else if floatValue, err := strconv.ParseFloat(strValue, 64); err == nil {
								litellmParams[key] = floatValue
							} else {
								// Keep as string
								litellmParams[key] = strValue
							}
						}
					}
				} else {
					// Apply existing conversion logic for non-JSON strings
					if strValue == "true" {
						litellmParams[key] = true
					} else if strValue == "false" {
						litellmParams[key] = false
					} else {
						// Try to convert numeric strings
						if intValue, err := strconv.Atoi(strValue); err == nil {
							litellmParams[key] = intValue
						} else if floatValue, err := strconv.ParseFloat(strValue, 64); err == nil {
							litellmParams[key] = floatValue
						} else {
							// Keep as string
							litellmParams[key] = strValue
						}
					}
				}
			} else {
				litellmParams[key] = value
			}
		}

		// Apply drop params at the end
		for _, paramToDrop := range dropParams {
			delete(litellmParams, paramToDrop)
		}
	}

	// Add litellm_credential_name to litellmParams if provided
	if credentialName := d.Get("litellm_credential_name").(string); credentialName != "" {
		litellmParams["litellm_credential_name"] = credentialName
	}
	cacheReadInputCost := configuredModelFloat(d, "cache_read_input_cost_per_million_tokens")
	if cacheReadInputCost != nil {
		*cacheReadInputCost /= 1000000.0
	}

	modelReq := ModelRequest{
		ModelName:     d.Get("model_name").(string),
		LiteLLMParams: litellmParams,
		ModelInfo: ModelInfoRequest{
			ID:                      modelID,
			DBModel:                 true,
			BaseModel:               pricingBaseModel,
			Tier:                    d.Get("tier").(string),
			Mode:                    d.Get("mode").(string),
			TeamID:                  d.Get("team_id").(string),
			MaxInputTokens:          configuredModelInt(d, "max_input_tokens"),
			MaxOutputTokens:         configuredModelInt(d, "max_output_tokens"),
			InputModalities:         configuredModelStringList(d, "input_modalities"),
			OutputModalities:        configuredModelStringList(d, "output_modalities"),
			SupportsReasoning:       configuredModelBool(d, "supports_reasoning"),
			SupportsFunctionCalling: configuredModelBool(d, "supports_function_calling"),
			SupportsVision:          configuredModelBool(d, "supports_vision"),
			InputCostPerCharacter:   configuredModelFloat(d, "input_cost_per_character"),
			CacheReadInputTokenCost: cacheReadInputCost,
			DefaultVoice:            configuredModelString(d, "default_voice"),
			ProbeLanguage:           configuredModelString(d, "probe_language"),
			ProbeText:               configuredModelString(d, "probe_text"),
			ProbeSkip:               configuredModelBool(d, "probe_skip"),
			MaxTokens:               configuredModelInt(d, "max_tokens"),
		},
		Additional: make(map[string]interface{}),
	}

	var resp *http.Response
	var err error
	if isUpdate {
		resp, err = MakeRequest(client, "PATCH", fmt.Sprintf(endpointModelUpdate, url.PathEscape(modelID)), modelReq)
	} else {
		resp, err = MakeRequest(client, "POST", endpointModelNew, modelReq)
	}
	if err != nil {
		return fmt.Errorf("failed to %s model: %w", map[bool]string{true: "update", false: "create"}[isUpdate], err)
	}
	defer resp.Body.Close()

	_, err = handleAPIResponse(resp, modelReq, client)
	if err != nil {
		if isUpdate && err.Error() == "model_not_found" {
			return createOrUpdateModel(d, m, false)
		}
		return fmt.Errorf("failed to %s model: %w", map[bool]string{true: "update", false: "create"}[isUpdate], err)
	}

	d.SetId(modelID)

	log.Printf("[INFO] Model created with ID %s. Starting retry mechanism to read the model...", modelID)
	// Read back the resource with retries to ensure the state is consistent
	return retryModelRead(d, m, 5)
}

func resourceLiteLLMModelCreate(d *schema.ResourceData, m interface{}) error {
	return createOrUpdateModel(d, m, false)
}

func resourceLiteLLMModelRead(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	query := url.Values{"modelId": []string{d.Id()}}
	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?%s", endpointModelInfo, query.Encode()), nil)
	if err != nil {
		return fmt.Errorf("failed to read model: %w", err)
	}
	defer resp.Body.Close()

	modelResp, err := handleModelInfoAPIResponse(resp, d.Id(), client)
	if err != nil {
		if err.Error() == "model_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to read model: %w", err)
	}

	// Update the state with values from the response or fall back to the data passed in during creation
	d.Set("model_name", GetStringValue(modelResp.ModelName, d.Get("model_name").(string)))
	d.Set("custom_llm_provider", GetStringValue(modelResp.LiteLLMParams.CustomLLMProvider, d.Get("custom_llm_provider").(string)))
	d.Set("tpm", GetIntValue(modelResp.LiteLLMParams.TPM, d.Get("tpm").(int)))
	d.Set("rpm", GetIntValue(modelResp.LiteLLMParams.RPM, d.Get("rpm").(int)))
	d.Set("model_api_base", GetStringValue(modelResp.LiteLLMParams.APIBase, d.Get("model_api_base").(string)))
	d.Set("api_version", GetStringValue(modelResp.LiteLLMParams.APIVersion, d.Get("api_version").(string)))
	// Routing and pricing keys are independent. For imports, reconstruct the
	// routing base from litellm_params.model by stripping exactly one matching
	// provider prefix. A model name containing further slashes stays intact.
	routingBaseModel := modelRoutingBase(modelResp.LiteLLMParams.CustomLLMProvider, modelResp.LiteLLMParams.Model)
	if routingBaseModel == "" {
		routingBaseModel = d.Get("base_model").(string)
	}
	if err := d.Set("base_model", routingBaseModel); err != nil {
		return fmt.Errorf("failed to set base_model: %w", err)
	}

	if modelResp.ModelInfo.BaseModel != nil {
		pricingBaseModel := *modelResp.ModelInfo.BaseModel
		if pricingBaseModel == routingBaseModel {
			pricingBaseModel = ""
		}
		if err := d.Set("pricing_base_model", pricingBaseModel); err != nil {
			return fmt.Errorf("failed to set pricing_base_model: %w", err)
		}
	}
	d.Set("tier", GetStringValue(modelResp.ModelInfo.Tier, d.Get("tier").(string)))
	d.Set("mode", GetStringValue(modelResp.ModelInfo.Mode, d.Get("mode").(string)))
	d.Set("team_id", GetStringValue(modelResp.ModelInfo.TeamID, d.Get("team_id").(string)))
	if modelResp.ModelInfo.MaxInputTokens != nil {
		if err := d.Set("max_input_tokens", *modelResp.ModelInfo.MaxInputTokens); err != nil {
			return fmt.Errorf("failed to set max_input_tokens: %w", err)
		}
	}
	if modelResp.ModelInfo.MaxOutputTokens != nil {
		if err := d.Set("max_output_tokens", *modelResp.ModelInfo.MaxOutputTokens); err != nil {
			return fmt.Errorf("failed to set max_output_tokens: %w", err)
		}
	}
	if modelResp.ModelInfo.InputModalities == nil {
		if err := d.Set("input_modalities", d.Get("input_modalities")); err != nil {
			return err
		}
	}
	if modelResp.ModelInfo.InputModalities != nil {
		if err := d.Set("input_modalities", *modelResp.ModelInfo.InputModalities); err != nil {
			return fmt.Errorf("failed to set input_modalities: %w", err)
		}
	}
	if modelResp.ModelInfo.OutputModalities == nil {
		if err := d.Set("output_modalities", d.Get("output_modalities")); err != nil {
			return err
		}
	}
	if modelResp.ModelInfo.OutputModalities != nil {
		if err := d.Set("output_modalities", *modelResp.ModelInfo.OutputModalities); err != nil {
			return fmt.Errorf("failed to set output_modalities: %w", err)
		}
	}
	if modelResp.ModelInfo.SupportsReasoning != nil {
		if err := d.Set("supports_reasoning", *modelResp.ModelInfo.SupportsReasoning); err != nil {
			return fmt.Errorf("failed to set supports_reasoning: %w", err)
		}
	}
	if modelResp.ModelInfo.SupportsFunctionCalling != nil {
		if err := d.Set("supports_function_calling", *modelResp.ModelInfo.SupportsFunctionCalling); err != nil {
			return fmt.Errorf("failed to set supports_function_calling: %w", err)
		}
	}
	if modelResp.ModelInfo.SupportsVision != nil {
		if err := d.Set("supports_vision", *modelResp.ModelInfo.SupportsVision); err != nil {
			return fmt.Errorf("failed to set supports_vision: %w", err)
		}
	}
	if modelResp.ModelInfo.InputCostPerCharacter != nil {
		if err := d.Set("input_cost_per_character", *modelResp.ModelInfo.InputCostPerCharacter); err != nil {
			return fmt.Errorf("failed to set input_cost_per_character: %w", err)
		}
	}
	if modelResp.ModelInfo.CacheReadInputTokenCost != nil {
		if err := d.Set("cache_read_input_cost_per_million_tokens", *modelResp.ModelInfo.CacheReadInputTokenCost*1000000.0); err != nil {
			return fmt.Errorf("failed to set cache_read_input_cost_per_million_tokens: %w", err)
		}
	}
	if modelResp.ModelInfo.DefaultVoice != nil {
		if err := d.Set("default_voice", *modelResp.ModelInfo.DefaultVoice); err != nil {
			return fmt.Errorf("failed to set default_voice: %w", err)
		}
	}
	if modelResp.ModelInfo.ProbeLanguage != nil {
		if err := d.Set("probe_language", *modelResp.ModelInfo.ProbeLanguage); err != nil {
			return fmt.Errorf("failed to set probe_language: %w", err)
		}
	}
	if modelResp.ModelInfo.ProbeText != nil {
		if err := d.Set("probe_text", *modelResp.ModelInfo.ProbeText); err != nil {
			return fmt.Errorf("failed to set probe_text: %w", err)
		}
	}
	if modelResp.ModelInfo.ProbeSkip != nil {
		if err := d.Set("probe_skip", *modelResp.ModelInfo.ProbeSkip); err != nil {
			return fmt.Errorf("failed to set probe_skip: %w", err)
		}
	}
	if modelResp.ModelInfo.MaxTokens != nil {
		if err := d.Set("max_tokens", *modelResp.ModelInfo.MaxTokens); err != nil {
			return fmt.Errorf("failed to set max_tokens: %w", err)
		}
	}

	// Preserve credential name from state since it might not be returned by API
	d.Set("litellm_credential_name", d.Get("litellm_credential_name").(string))

	// Store sensitive information
	d.Set("model_api_key", d.Get("model_api_key"))
	d.Set("aws_access_key_id", d.Get("aws_access_key_id"))
	d.Set("aws_secret_access_key", d.Get("aws_secret_access_key"))
	d.Set("aws_region_name", GetStringValue(modelResp.LiteLLMParams.AWSRegionName, d.Get("aws_region_name").(string)))
	d.Set("aws_session_name", d.Get("aws_session_name"))
	d.Set("aws_role_name", d.Get("aws_role_name"))

	// Store cost information
	d.Set("input_cost_per_million_tokens", d.Get("input_cost_per_million_tokens"))
	d.Set("output_cost_per_million_tokens", d.Get("output_cost_per_million_tokens"))

	// Handle thinking configuration
	if _, ok := d.GetOk("thinking_enabled"); ok {
		// Keep the existing value from state
		thinkingEnabled := d.Get("thinking_enabled").(bool)
		d.Set("thinking_enabled", thinkingEnabled)

		// Only set thinking_budget_tokens if thinking is enabled and we have a value in state
		if thinkingEnabled {
			if _, ok := d.GetOk("thinking_budget_tokens"); ok {
				d.Set("thinking_budget_tokens", d.Get("thinking_budget_tokens").(int))
			}
		}
	} else {
		// Fall back to API response if no state value exists
		if modelResp.LiteLLMParams.Thinking != nil {
			if thinkingType, ok := modelResp.LiteLLMParams.Thinking["type"].(string); ok && thinkingType == "enabled" {
				d.Set("thinking_enabled", true)
				if budgetTokens, ok := modelResp.LiteLLMParams.Thinking["budget_tokens"].(float64); ok {
					d.Set("thinking_budget_tokens", int(budgetTokens))
				}
			} else {
				d.Set("thinking_enabled", false)
			}
		} else {
			d.Set("thinking_enabled", false)
		}
	}

	// Handle merge_reasoning_content_in_choices - preserve state value if not returned by API
	if _, ok := d.GetOk("merge_reasoning_content_in_choices"); ok {
		// Keep the existing value from state
		d.Set("merge_reasoning_content_in_choices", d.Get("merge_reasoning_content_in_choices").(bool))
	} else {
		// Only set from API response if we don't have a value in state
		d.Set("merge_reasoning_content_in_choices", modelResp.LiteLLMParams.MergeReasoningContentInChoices)
	}

	// Preserve additional_litellm_params from state since API might not return all custom parameters
	if _, ok := d.GetOk("additional_litellm_params"); ok {
		d.Set("additional_litellm_params", d.Get("additional_litellm_params"))
	}

	return nil
}

func resourceLiteLLMModelUpdate(d *schema.ResourceData, m interface{}) error {
	return createOrUpdateModel(d, m, true)
}

func resourceLiteLLMModelDelete(d *schema.ResourceData, m interface{}) error {
	client, ok := m.(*Client)
	if !ok {
		return fmt.Errorf("invalid type assertion for client")
	}

	deleteReq := struct {
		ID string `json:"id"`
	}{
		ID: d.Id(),
	}

	resp, err := MakeRequest(client, "POST", endpointModelDelete, deleteReq)
	if err != nil {
		return fmt.Errorf("failed to delete model: %w", err)
	}
	defer resp.Body.Close()

	_, err = handleAPIResponse(resp, deleteReq, client)
	if err != nil {
		if err.Error() == "model_not_found" {
			d.SetId("")
			return nil
		}
		return fmt.Errorf("failed to delete model: %w", err)
	}

	d.SetId("")
	return nil
}
