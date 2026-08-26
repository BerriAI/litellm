package litellm

import (
	"reflect"
	"testing"
)

func TestMergeAdditionalLiteLLMParams_PassesAdditionalDropParams(t *testing.T) {
	litellmParams := map[string]interface{}{
		"model":               "azure/gpt-5-mini",
		"custom_llm_provider": "azure",
		"drop_params":         true,
	}
	additional := map[string]interface{}{
		"additional_drop_params": `["reasoning_content", "include_reasoning", "promptCacheKey"]`,
		"drop_params":            "true",
		"timeout":                "60",
		"temperature_scale":      "0.75",
		"note":                   "keep-as-string",
		"complex_config":         `{"nested":{"value":42}}`,
	}

	mergeAdditionalLiteLLMParams(litellmParams, additional)

	got, ok := litellmParams["additional_drop_params"].([]string)
	if !ok {
		t.Fatalf("additional_drop_params type = %T, want []string", litellmParams["additional_drop_params"])
	}
	want := []string{"reasoning_content", "include_reasoning", "promptCacheKey"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("additional_drop_params = %#v, want %#v", got, want)
	}

	if _, stillThere := litellmParams["model"]; !stillThere {
		t.Fatalf("model was deleted from litellmParams; additional_drop_params must not mutate sibling keys")
	}
	if litellmParams["drop_params"] != true {
		t.Fatalf("drop_params = %#v, want true", litellmParams["drop_params"])
	}
	if litellmParams["timeout"] != 60 {
		t.Fatalf("timeout = %#v, want 60", litellmParams["timeout"])
	}
	if litellmParams["temperature_scale"] != 0.75 {
		t.Fatalf("temperature_scale = %#v, want 0.75", litellmParams["temperature_scale"])
	}
	if litellmParams["note"] != "keep-as-string" {
		t.Fatalf("note = %#v, want keep-as-string", litellmParams["note"])
	}
	cfg, ok := litellmParams["complex_config"].(map[string]interface{})
	if !ok {
		t.Fatalf("complex_config type = %T, want map", litellmParams["complex_config"])
	}
	nested, ok := cfg["nested"].(map[string]interface{})
	if !ok || nested["value"] != float64(42) {
		t.Fatalf("complex_config = %#v, want nested.value=42", litellmParams["complex_config"])
	}
}

func TestMergeAdditionalLiteLLMParams_DoesNotDeleteNamedKeys(t *testing.T) {
	litellmParams := map[string]interface{}{
		"model":               "x",
		"reasoning_effort":    "high",
		"reasoningEffort":     "high",
		"custom_llm_provider": "azure",
	}
	additional := map[string]interface{}{
		"additional_drop_params": `["reasoningEffort", "reasoning_effort"]`,
	}

	mergeAdditionalLiteLLMParams(litellmParams, additional)

	if _, ok := litellmParams["reasoning_effort"]; !ok {
		t.Fatal("reasoning_effort was deleted from payload")
	}
	if _, ok := litellmParams["reasoningEffort"]; !ok {
		t.Fatal("reasoningEffort was deleted from payload")
	}
	got := litellmParams["additional_drop_params"].([]string)
	want := []string{"reasoningEffort", "reasoning_effort"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("additional_drop_params = %#v, want %#v", got, want)
	}
}
