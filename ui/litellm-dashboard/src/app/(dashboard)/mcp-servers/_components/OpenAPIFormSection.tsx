import { Info } from "lucide-react";
import React, { useState } from "react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Input } from "@/components/ui/input";
import { AUTH_TYPE, OAUTH_FLOW } from "@/components/mcp_tools/types";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { requiredRule } from "@/components/common_components/formRules";
import OpenAPIQuickPicker, { OpenAPIRegistryEntry, OpenAPIKeyTool } from "./OpenAPIQuickPicker";
import { McpForm, resetFields, setFieldsValue } from "./mcpFormStore";
import { textControl } from "./mcpFieldRules";

interface OpenAPIFormSectionProps {
  form: McpForm;
  accessToken: string | null;
  /** Called when a preset is selected so the parent can sync its formValues state. */
  onValuesChange: (updates: Record<string, any>) => void;
  /** Called when key tools change (from registry preset selection). */
  onKeyToolsChange?: (tools: OpenAPIKeyTool[]) => void;
  /** Called when a preset is selected so the parent can set the logo URL from icon_url. */
  onLogoUrlChange?: (url: string | undefined) => void;
  /** Called when the OAuth docs URL changes (e.g. link to create a GitHub OAuth App). */
  onOAuthDocsUrlChange?: (url: string | null) => void;
}

/**
 * Encapsulates all OpenAPI-specific form fields:
 *  - popular API quick-picker (logos)
 *  - spec URL input
 */
const OpenAPIFormSection: React.FC<OpenAPIFormSectionProps> = ({
  form,
  accessToken,
  onValuesChange,
  onKeyToolsChange,
  onLogoUrlChange,
  onOAuthDocsUrlChange,
}) => {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const handlePresetSelect = (entry: OpenAPIRegistryEntry) => {
    setSelectedPreset(entry.name);
    onKeyToolsChange?.(entry.key_tools ?? []);
    onLogoUrlChange?.(entry.icon_url || undefined);
    const updates: Record<string, any> = {
      spec_path: entry.spec_url,
    };
    if (entry.oauth) {
      updates.auth_type = AUTH_TYPE.OAUTH2;
      // OAuth2 registry entries always use the interactive (PKCE) flow — users
      // authorize via their browser, not machine-to-machine client credentials.
      updates.oauth_flow_type = OAUTH_FLOW.INTERACTIVE;
      updates.authorization_url = entry.oauth.authorization_url;
      updates.token_url = entry.oauth.token_url;
      setFieldsValue(form, updates);
      onOAuthDocsUrlChange?.(entry.oauth.docs_url ?? null);
    } else {
      resetFields(form, ["auth_type", "authorization_url", "token_url"]);
      setFieldsValue(form, updates);
      onOAuthDocsUrlChange?.(null);
    }
    onValuesChange(updates);
  };

  return (
    <>
      <OpenAPIQuickPicker accessToken={accessToken} selectedName={selectedPreset} onSelect={handlePresetSelect} />

      <MountedFormField
        label={
          <span className="text-sm font-medium text-foreground flex items-center">
            OpenAPI Spec URL
            <SimpleTooltip content="URL to an OpenAPI specification (JSON or YAML). MCP tools will be automatically generated from the API endpoints defined in the spec.">
              <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
            </SimpleTooltip>
          </span>
        }
        name="spec_path"
        required
        rules={{ validate: { required: requiredRule("Please enter an OpenAPI spec URL") } }}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
            className="rounded-lg border-border focus:border-info focus:ring-ring"
            onChange={(event) => {
              control.onChange(event);
              // Clear the preset selection when the user manually edits the spec URL
              // so stale suggested tools from a previous preset don't persist.
              setSelectedPreset(null);
              onKeyToolsChange?.([]);
              onOAuthDocsUrlChange?.(null);
            }}
          />
        )}
      </MountedFormField>
    </>
  );
};

export default OpenAPIFormSection;
export type { OpenAPIKeyTool };
