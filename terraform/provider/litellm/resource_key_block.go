package litellm

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"

	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
)

const (
	endpointKeyBlock   = "/key/block"
	endpointKeyUnblock = "/key/unblock"
)

type KeyBlockInfoResponse struct {
	Info struct {
		Blocked *bool `json:"blocked"`
	} `json:"info"`
}

func resourceLiteLLMKeyBlock() *schema.Resource {
	return &schema.Resource{
		Create: resourceLiteLLMKeyBlockCreate,
		Read:   resourceLiteLLMKeyBlockRead,
		Delete: resourceLiteLLMKeyBlockDelete,

		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},

		Schema: map[string]*schema.Schema{
			"key": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Sensitive:   true,
				Description: "The API key to block, as the raw sk- value or its SHA-256 token hash. Destroying this resource unblocks the key",
				DiffSuppressFunc: func(k, old, new string, d *schema.ResourceData) bool {
					return old != "" && hashedKeyToken(old) == hashedKeyToken(new)
				},
			},
			"blocked": {
				Type:        schema.TypeBool,
				Computed:    true,
				Description: "Whether the key is currently blocked",
			},
		},
	}
}

func resourceLiteLLMKeyBlockCreate(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	// Block by the SHA-256 token hash so the raw key never appears in the
	// request, the resource ID, or Terraform plan output.
	token := hashedKeyToken(d.Get("key").(string))

	log.Printf("[INFO] Blocking key")

	resp, err := MakeRequest(client, "POST", endpointKeyBlock, map[string]interface{}{"key": token})
	if err != nil {
		return fmt.Errorf("error blocking key: %w", err)
	}
	defer resp.Body.Close()

	if err := handleResponse(resp, "blocking key"); err != nil {
		return err
	}

	d.SetId(token)
	return resourceLiteLLMKeyBlockRead(d, m)
}

func resourceLiteLLMKeyBlockRead(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)
	key := d.Id()

	resp, err := MakeRequest(client, "GET", fmt.Sprintf("/key/info?key=%s", url.QueryEscape(key)), nil)
	if err != nil {
		return fmt.Errorf("error reading key info: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		log.Printf("[WARN] Key not found, removing key block from state")
		d.SetId("")
		return nil
	}

	if err := handleResponse(resp, "reading key info"); err != nil {
		return err
	}

	var infoResp KeyBlockInfoResponse
	if err := json.NewDecoder(resp.Body).Decode(&infoResp); err != nil {
		return fmt.Errorf("error decoding key info response: %w", err)
	}

	if infoResp.Info.Blocked == nil || !*infoResp.Info.Blocked {
		log.Printf("[WARN] Key is no longer blocked, removing key block from state")
		d.SetId("")
		return nil
	}

	// Keep the configured key value; only fill it from the hashed ID when
	// importing, where no configured value exists yet.
	if _, ok := d.GetOk("key"); !ok {
		d.Set("key", key)
	}
	d.Set("blocked", true)
	return nil
}

func resourceLiteLLMKeyBlockDelete(d *schema.ResourceData, m interface{}) error {
	client := m.(*Client)

	log.Printf("[INFO] Unblocking key")

	resp, err := MakeRequest(client, "POST", endpointKeyUnblock, map[string]interface{}{"key": d.Id()})
	if err != nil {
		return fmt.Errorf("error unblocking key: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusNotFound {
		if err := handleResponse(resp, "unblocking key"); err != nil {
			return err
		}
	}

	d.SetId("")
	return nil
}
