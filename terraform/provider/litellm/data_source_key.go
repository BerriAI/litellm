package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"strconv"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointKeyInfo = "/key/info"
	endpointKeyList = "/key/list"
)

type keyInfoDetail struct {
	Token               string                 `json:"token"`
	KeyName             string                 `json:"key_name"`
	KeyAlias            string                 `json:"key_alias"`
	Spend               float64                `json:"spend"`
	MaxBudget           *float64               `json:"max_budget"`
	Models              []string               `json:"models"`
	UserID              string                 `json:"user_id"`
	TeamID              string                 `json:"team_id"`
	OrgID               string                 `json:"org_id"`
	TPMLimit            *int                   `json:"tpm_limit"`
	RPMLimit            *int                   `json:"rpm_limit"`
	MaxParallelRequests *int                   `json:"max_parallel_requests"`
	BudgetDuration      string                 `json:"budget_duration"`
	Metadata            map[string]interface{} `json:"metadata"`
	Blocked             *bool                  `json:"blocked"`
	Expires             string                 `json:"expires"`
	CreatedAt           string                 `json:"created_at"`
	UpdatedAt           string                 `json:"updated_at"`
}

type keyInfoEnvelope struct {
	Key  string        `json:"key"`
	Info keyInfoDetail `json:"info"`
}

type keyListEnvelope struct {
	Keys        []keyInfoDetail `json:"keys"`
	TotalCount  int             `json:"total_count"`
	CurrentPage int             `json:"current_page"`
	TotalPages  int             `json:"total_pages"`
}

func dataSourceLiteLLMKey() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMKeyRead,

		Schema: map[string]*schema.Schema{
			"key": {
				Type:        schema.TypeString,
				Required:    true,
				Sensitive:   true,
				Description: "The API key (or its hash) to look up",
			},
			"token_id": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Hashed token identifier of the key",
			},
			"key_name": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Redacted display name of the key",
			},
			"key_alias": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"models": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"spend": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"max_budget": {
				Type:     schema.TypeFloat,
				Computed: true,
			},
			"user_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"team_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"organization_id": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"tpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"rpm_limit": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"max_parallel_requests": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"budget_duration": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"metadata": {
				Type:     schema.TypeMap,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"tags": {
				Type:     schema.TypeList,
				Computed: true,
				Elem:     &schema.Schema{Type: schema.TypeString},
			},
			"blocked": {
				Type:     schema.TypeBool,
				Computed: true,
			},
			"expires": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"created_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
			"updated_at": {
				Type:     schema.TypeString,
				Computed: true,
			},
		},
	}
}

func dataSourceLiteLLMKeyRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	// Look up by the SHA-256 token hash so the raw key never appears in the
	// request URL, where reverse-proxy access logs could record it.
	key := hashedKeyToken(d.Get("key").(string))

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?key=%s", endpointKeyInfo, url.QueryEscape(key)), nil)
	if err != nil {
		return fmt.Errorf("failed to read key info: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "reading key info"); err != nil {
		return err
	}

	var envelope keyInfoEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("failed to decode key info response: %w", err)
	}
	info := envelope.Info

	// Never persist the raw key as the ID; the hashed token is safe to store.
	d.SetId(GetStringValue(info.Token, "key"))
	d.Set("token_id", info.Token)
	d.Set("key_name", info.KeyName)
	d.Set("key_alias", info.KeyAlias)
	d.Set("models", info.Models)
	d.Set("spend", info.Spend)
	if info.MaxBudget != nil {
		d.Set("max_budget", *info.MaxBudget)
	}
	d.Set("user_id", info.UserID)
	d.Set("team_id", info.TeamID)
	d.Set("organization_id", info.OrgID)
	if info.TPMLimit != nil {
		d.Set("tpm_limit", *info.TPMLimit)
	}
	if info.RPMLimit != nil {
		d.Set("rpm_limit", *info.RPMLimit)
	}
	if info.MaxParallelRequests != nil {
		d.Set("max_parallel_requests", *info.MaxParallelRequests)
	}
	d.Set("budget_duration", info.BudgetDuration)

	metadata := map[string]string{}
	for k, v := range info.Metadata {
		if s, ok := v.(string); ok {
			metadata[k] = s
		}
	}
	d.Set("metadata", metadata)
	d.Set("tags", toStringSlice(info.Metadata["tags"]))

	if info.Blocked != nil {
		d.Set("blocked", *info.Blocked)
	}
	d.Set("expires", info.Expires)
	d.Set("created_at", info.CreatedAt)
	d.Set("updated_at", info.UpdatedAt)

	log.Printf("[INFO] Successfully read key info for token: %s", info.Token)
	return nil
}

func dataSourceLiteLLMKeys() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMKeysRead,

		Schema: map[string]*schema.Schema{
			"page": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     1,
				Description: "Page number for pagination",
			},
			"size": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     100,
				Description: "Number of keys per page",
			},
			"user_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter keys by user ID",
			},
			"team_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter keys by team ID",
			},
			"organization_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter keys by organization ID",
			},
			"key_alias": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter keys by key alias",
			},
			"include_team_keys": {
				Type:        schema.TypeBool,
				Optional:    true,
				Description: "Include all keys for teams the caller is an admin of",
			},
			"total_count": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"total_pages": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"current_page": {
				Type:     schema.TypeInt,
				Computed: true,
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Hashed token identifiers of the returned keys",
			},
			"keys": {
				Type:     schema.TypeList,
				Computed: true,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"token_id":        {Type: schema.TypeString, Computed: true},
						"key_name":        {Type: schema.TypeString, Computed: true},
						"key_alias":       {Type: schema.TypeString, Computed: true},
						"spend":           {Type: schema.TypeFloat, Computed: true},
						"max_budget":      {Type: schema.TypeFloat, Computed: true},
						"models":          {Type: schema.TypeList, Computed: true, Elem: &schema.Schema{Type: schema.TypeString}},
						"user_id":         {Type: schema.TypeString, Computed: true},
						"team_id":         {Type: schema.TypeString, Computed: true},
						"organization_id": {Type: schema.TypeString, Computed: true},
						"tpm_limit":       {Type: schema.TypeInt, Computed: true},
						"rpm_limit":       {Type: schema.TypeInt, Computed: true},
						"budget_duration": {Type: schema.TypeString, Computed: true},
						"blocked":         {Type: schema.TypeBool, Computed: true},
						"expires":         {Type: schema.TypeString, Computed: true},
						"created_at":      {Type: schema.TypeString, Computed: true},
						"updated_at":      {Type: schema.TypeString, Computed: true},
					},
				},
			},
		},
	}
}

func dataSourceLiteLLMKeysRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	query := url.Values{}
	query.Set("return_full_object", "true")
	query.Set("page", strconv.Itoa(d.Get("page").(int)))
	query.Set("size", strconv.Itoa(d.Get("size").(int)))
	for param, attr := range map[string]string{
		"user_id":         "user_id",
		"team_id":         "team_id",
		"organization_id": "organization_id",
		"key_alias":       "key_alias",
	} {
		if v, ok := d.GetOk(attr); ok {
			query.Set(param, v.(string))
		}
	}
	if d.Get("include_team_keys").(bool) {
		query.Set("include_team_keys", "true")
	}

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?%s", endpointKeyList, query.Encode()), nil)
	if err != nil {
		return fmt.Errorf("failed to list keys: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing keys"); err != nil {
		return err
	}

	var envelope keyListEnvelope
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("failed to decode key list response: %w", err)
	}

	ids := make([]string, 0, len(envelope.Keys))
	keys := make([]map[string]interface{}, 0, len(envelope.Keys))
	for _, k := range envelope.Keys {
		ids = append(ids, k.Token)
		keys = append(keys, map[string]interface{}{
			"token_id":        k.Token,
			"key_name":        k.KeyName,
			"key_alias":       k.KeyAlias,
			"spend":           k.Spend,
			"max_budget":      keyDerefFloat(k.MaxBudget),
			"models":          k.Models,
			"user_id":         k.UserID,
			"team_id":         k.TeamID,
			"organization_id": k.OrgID,
			"tpm_limit":       keyDerefInt(k.TPMLimit),
			"rpm_limit":       keyDerefInt(k.RPMLimit),
			"budget_duration": k.BudgetDuration,
			"blocked":         k.Blocked != nil && *k.Blocked,
			"expires":         k.Expires,
			"created_at":      k.CreatedAt,
			"updated_at":      k.UpdatedAt,
		})
	}

	d.SetId(query.Encode())
	d.Set("total_count", envelope.TotalCount)
	d.Set("total_pages", envelope.TotalPages)
	d.Set("current_page", envelope.CurrentPage)
	d.Set("ids", ids)
	d.Set("keys", keys)

	log.Printf("[INFO] Successfully listed %d keys", len(keys))
	return nil
}

func keyDerefFloat(v *float64) float64 {
	if v == nil {
		return 0
	}
	return *v
}

func keyDerefInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
