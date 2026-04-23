import React, { useState } from "react";
import { Form, Input } from "antd";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { FormInstance } from "antd/es/form";
import { AUTH_TYPE, OAUTH_FLOW } from "./types";
import OpenAPIQuickPicker, {
  OpenAPIRegistryEntry,
  OpenAPIKeyTool,
} from "./OpenAPIQuickPicker";

interface OpenAPIFormSectionProps {
  form: FormInstance;
  accessToken: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onValuesChange: (updates: Record<string, any>) => void;
  onKeyToolsChange?: (tools: OpenAPIKeyTool[]) => void;
  onLogoUrlChange?: (url: string | undefined) => void;
  onOAuthDocsUrlChange?: (url: string | null) => void;
}

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
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const updates: Record<string, any> = {
      spec_path: entry.spec_url,
    };
    if (entry.oauth) {
      updates.auth_type = AUTH_TYPE.OAUTH2;
      updates.oauth_flow_type = OAUTH_FLOW.INTERACTIVE;
      updates.authorization_url = entry.oauth.authorization_url;
      updates.token_url = entry.oauth.token_url;
      form.setFieldsValue(updates);
      onOAuthDocsUrlChange?.(entry.oauth.docs_url ?? null);
    } else {
      form.resetFields(["auth_type", "authorization_url", "token_url"]);
      form.setFieldsValue(updates);
      onOAuthDocsUrlChange?.(null);
    }
    onValuesChange(updates);
  };

  return (
    <>
      <OpenAPIQuickPicker
        accessToken={accessToken}
        selectedName={selectedPreset}
        onSelect={handlePresetSelect}
      />

      <Form.Item
        label={
          <span className="text-sm font-medium text-foreground flex items-center">
            OpenAPI Spec URL
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="ml-2 h-3 w-3 text-primary cursor-help" />
                </TooltipTrigger>
                <TooltipContent className="max-w-xs">
                  URL to an OpenAPI specification (JSON or YAML). MCP tools
                  will be automatically generated from the API endpoints
                  defined in the spec.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </span>
        }
        name="spec_path"
        rules={[
          { required: true, message: "Please enter an OpenAPI spec URL" },
        ]}
      >
        <Input
          placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
          className="rounded-lg"
          onChange={() => {
            setSelectedPreset(null);
            onKeyToolsChange?.([]);
            onOAuthDocsUrlChange?.(null);
          }}
        />
      </Form.Item>
    </>
  );
};

export default OpenAPIFormSection;
export type { OpenAPIKeyTool };
