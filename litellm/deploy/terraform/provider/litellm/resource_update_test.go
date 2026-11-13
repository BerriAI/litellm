package litellm

import (
	"context"
	"testing"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/terraform"
)

// Build a real SDK update diff rather than making unchanged controls look like
// values being created from empty state with TestResourceDataRaw alone.
func resourceUpdateData(t *testing.T, resource *schema.Resource, id string, before, after map[string]interface{}) *schema.ResourceData {
	t.Helper()
	prior := schema.TestResourceDataRaw(t, resource.Schema, before)
	prior.SetId(id)
	diff, err := resource.Diff(context.Background(), prior.State(), terraform.NewResourceConfigRaw(after), nil)
	if err != nil {
		t.Fatalf("building resource diff: %v", err)
	}
	data, err := schema.InternalMap(resource.Schema).Data(prior.State(), diff)
	if err != nil {
		t.Fatalf("building resource data: %v", err)
	}
	return data
}
