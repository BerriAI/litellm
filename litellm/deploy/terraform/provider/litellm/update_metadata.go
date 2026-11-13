package litellm

import (
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

// Read metadata immediately before an update, not from the saved plan. JSON
// numbers stay lossless and foreign maps, arrays and booleans remain untouched.
func currentResourceMetadata(client *Client, infoEndpoint, idParameter, resourceID, envelope string) (map[string]interface{}, error) {
	query := url.Values{idParameter: []string{resourceID}}
	response, err := MakeRequest(client, "GET", infoEndpoint+"?"+query.Encode(), nil)
	if err != nil {
		return nil, fmt.Errorf("reading current metadata: %w", err)
	}
	defer response.Body.Close()
	if err := handleResponse(response, "reading current metadata"); err != nil {
		return nil, err
	}

	var info map[string]interface{}
	decoder := json.NewDecoder(response.Body)
	decoder.UseNumber()
	if err := decoder.Decode(&info); err != nil {
		return nil, fmt.Errorf("decoding current metadata: %w", err)
	}
	if envelope != "" {
		nested, ok := info[envelope].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("current metadata response has no %s object", envelope)
		}
		info = nested
	}
	if info["metadata"] == nil {
		return make(map[string]interface{}), nil
	}
	metadata, ok := info["metadata"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("current metadata is not a JSON object")
	}
	return metadata, nil
}

// A metadata attribute is string-valued, but its update is a key-level delta.
// Unchanged saved values must never overwrite newer server-owned metadata.
func mergeChangedMetadata(d *schema.ResourceData, metadata map[string]interface{}) {
	if !d.HasChange("metadata") {
		return
	}
	oldValue, newValue := d.GetChange("metadata")
	oldMetadata, _ := oldValue.(map[string]interface{})
	newMetadata, _ := newValue.(map[string]interface{})
	for key := range oldMetadata {
		if _, retained := newMetadata[key]; !retained {
			delete(metadata, key)
		}
	}
	for key, value := range newMetadata {
		old, existed := oldMetadata[key]
		if !existed || old != value {
			metadata[key] = value
		}
	}
}
