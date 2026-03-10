import React, { useState } from "react";
import { Form, Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { FormInstance } from "antd/es/form";
import { AUTH_TYPE } from "./types";
import OpenAPIQuickPicker, { OpenAPIRegistryEntry, OpenAPIKeyTool } from "./OpenAPIQuickPicker";

interface OpenAPIFormSectionProps {
  form: FormInstance;
  accessToken: string | null;
  /** Called when a preset is selected so the parent can sync its formValues state. */
  onValuesChange: (updates: Record<string, any>) => void;
  /** Called when key tools change (from registry preset selection). */
  onKeyToolsChange?: (tools: OpenAPIKeyTool[]) => void;
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
}) => {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const handlePresetSelect = (entry: OpenAPIRegistryEntry) => {
    setSelectedPreset(entry.name);
    onKeyToolsChange?.(entry.key_tools ?? []);
    const updates: Record<string, any> = {
      spec_path: entry.spec_url,
    };
    if (entry.oauth) {
      updates.auth_type = AUTH_TYPE.OAUTH2;
      updates.authorization_url = entry.oauth.authorization_url;
      updates.token_url = entry.oauth.token_url;
      form.setFieldsValue(updates);
    } else {
      // resetFields is required to visually clear Ant Design form fields —
      // setFieldsValue with undefined silently skips undefined keys.
      form.resetFields(["auth_type", "authorization_url", "token_url"]);
      form.setFieldsValue(updates);
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
          <span className="text-sm font-medium text-gray-700 flex items-center">
            OpenAPI Spec URL
            <Tooltip title="URL to an OpenAPI specification (JSON or YAML). MCP tools will be automatically generated from the API endpoints defined in the spec.">
              <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
            </Tooltip>
          </span>
        }
        name="spec_path"
        rules={[{ required: true, message: "Please enter an OpenAPI spec URL" }]}
      >
        <Input
          placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
          onChange={() => {
            // Clear the preset selection when the user manually edits the spec URL
            // so stale suggested tools from a previous preset don't persist.
            setSelectedPreset(null);
            onKeyToolsChange?.([]);
          }}
        />
      </Form.Item>
    </>
  );
};

export default OpenAPIFormSection;
export type { OpenAPIKeyTool };
