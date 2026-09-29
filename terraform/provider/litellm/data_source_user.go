package litellm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const endpointUserList = "/user/list"

func dataSourceLiteLLMUser() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMUserRead,

		Schema: map[string]*schema.Schema{
			"user_id": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "ID of the user to retrieve",
			},
			"user_email": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Email address of the user",
			},
			"user_alias": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Descriptive name for the user",
			},
			"user_role": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Role of the user on the proxy",
			},
			"teams": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "List of team IDs the user belongs to",
			},
			"models": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Models the user is allowed to call",
			},
			"max_budget": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Maximum budget in USD for the user",
			},
			"spend": {
				Type:        schema.TypeFloat,
				Computed:    true,
				Description: "Current spend in USD for the user",
			},
			"budget_duration": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Budget reset period for the user",
			},
			"tpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Tokens per minute limit for the user",
			},
			"rpm_limit": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Requests per minute limit for the user",
			},
			"max_parallel_requests": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Maximum number of parallel requests for the user",
			},
			"metadata": {
				Type:        schema.TypeMap,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "Metadata for the user",
			},
			"model_max_budget": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "JSON string of per-model budget config",
			},
		},
	}
}

func dataSourceLiteLLMUserRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	userID := d.Get("user_id").(string)

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?user_id=%s", endpointUserInfo, url.QueryEscape(userID)), nil)
	if err != nil {
		return fmt.Errorf("failed to read user: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return fmt.Errorf("user '%s' not found", userID)
	}

	if err := handleResponse(resp, "reading user"); err != nil {
		return err
	}

	var infoResp userInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&infoResp); err != nil {
		return fmt.Errorf("error decoding user info response: %w", err)
	}
	if infoResp.UserInfo == nil {
		return fmt.Errorf("user '%s' not found", userID)
	}

	d.SetId(userID)
	setUserStateFromInfo(d, infoResp.UserInfo)
	if v, ok := infoResp.UserInfo["spend"].(float64); ok {
		d.Set("spend", v)
	}

	return nil
}

func dataSourceLiteLLMUsers() *schema.Resource {
	return &schema.Resource{
		Read: dataSourceLiteLLMUsersRead,

		Schema: map[string]*schema.Schema{
			"role": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter users by role",
			},
			"user_ids": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Comma-separated list of user IDs to filter by",
			},
			"user_email": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter users by partial email match",
			},
			"team": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Filter users by team ID",
			},
			"page": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     1,
				Description: "Page number to fetch",
			},
			"page_size": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     25,
				Description: "Number of users per page (max 100)",
			},
			"sort_by": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Column to sort by (e.g. 'user_id', 'user_email', 'created_at')",
			},
			"sort_order": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Sort order, 'asc' or 'desc'",
			},
			"users": {
				Type:        schema.TypeList,
				Computed:    true,
				Description: "Users returned for the requested page",
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"user_id":    {Type: schema.TypeString, Computed: true},
						"user_email": {Type: schema.TypeString, Computed: true},
						"user_alias": {Type: schema.TypeString, Computed: true},
						"user_role":  {Type: schema.TypeString, Computed: true},
						"teams": {
							Type:     schema.TypeList,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"models": {
							Type:     schema.TypeList,
							Computed: true,
							Elem:     &schema.Schema{Type: schema.TypeString},
						},
						"max_budget": {Type: schema.TypeFloat, Computed: true},
						"spend":      {Type: schema.TypeFloat, Computed: true},
						"tpm_limit":  {Type: schema.TypeInt, Computed: true},
						"rpm_limit":  {Type: schema.TypeInt, Computed: true},
						"key_count":  {Type: schema.TypeInt, Computed: true},
						"created_at": {Type: schema.TypeString, Computed: true},
					},
				},
			},
			"ids": {
				Type:        schema.TypeList,
				Computed:    true,
				Elem:        &schema.Schema{Type: schema.TypeString},
				Description: "IDs of the users returned for the requested page",
			},
			"total": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Total number of users matching the filters",
			},
			"total_pages": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Total number of pages available",
			},
		},
	}
}

type userListResponse struct {
	Users      []map[string]interface{} `json:"users"`
	Total      int                      `json:"total"`
	TotalPages int                      `json:"total_pages"`
}

func userListQuery(d *schema.ResourceData) string {
	query := url.Values{}
	for _, key := range []string{"role", "user_ids", "user_email", "team", "sort_by", "sort_order"} {
		if v, ok := d.GetOk(key); ok {
			query.Set(key, v.(string))
		}
	}
	query.Set("page", strconv.Itoa(d.Get("page").(int)))
	query.Set("page_size", strconv.Itoa(d.Get("page_size").(int)))
	return query.Encode()
}

func userListEntry(user map[string]interface{}) map[string]interface{} {
	entry := map[string]interface{}{}
	for _, key := range []string{"user_id", "user_email", "user_alias", "user_role", "created_at"} {
		if v, ok := user[key].(string); ok {
			entry[key] = v
		}
	}
	for _, key := range []string{"max_budget", "spend"} {
		if v, ok := user[key].(float64); ok {
			entry[key] = v
		}
	}
	for _, key := range []string{"tpm_limit", "rpm_limit", "key_count"} {
		if v, ok := user[key].(float64); ok {
			entry[key] = int(v)
		}
	}
	for _, key := range []string{"teams", "models"} {
		if v, ok := user[key].([]interface{}); ok {
			entry[key] = v
		}
	}
	return entry
}

func dataSourceLiteLLMUsersRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	query := userListQuery(d)
	resp, err := MakeRequest(client, "GET", fmt.Sprintf("%s?%s", endpointUserList, query), nil)
	if err != nil {
		return fmt.Errorf("failed to list users: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "listing users"); err != nil {
		return err
	}

	var listResp userListResponse
	if err := json.NewDecoder(resp.Body).Decode(&listResp); err != nil {
		return fmt.Errorf("error decoding user list response: %w", err)
	}

	users := make([]map[string]interface{}, 0, len(listResp.Users))
	ids := make([]string, 0, len(listResp.Users))
	for _, user := range listResp.Users {
		entry := userListEntry(user)
		if id, ok := entry["user_id"].(string); ok {
			ids = append(ids, id)
		}
		users = append(users, entry)
	}

	d.SetId(fmt.Sprintf("users?%s", query))
	d.Set("users", users)
	d.Set("ids", ids)
	d.Set("total", listResp.Total)
	d.Set("total_pages", listResp.TotalPages)

	return nil
}
