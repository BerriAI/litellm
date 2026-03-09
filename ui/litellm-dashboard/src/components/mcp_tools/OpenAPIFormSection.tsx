import React, { useState } from "react";
import { Form, Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { FormInstance } from "antd/es/form";
import { AUTH_TYPE } from "./types";
import OpenAPIQuickPicker, { OpenAPIRegistryEntry } from "./OpenAPIQuickPicker";

interface OpenAPIFormSectionProps {
  form: FormInstance;
  accessToken: string | null;
  /** Called when a preset is selected so the parent can sync its formValues state. */
  onValuesChange: (updates: Record<string, any>) => void;
}

/**
 * Encapsulates all OpenAPI-specific form fields:
 *  - popular API quick-picker (logos)
 *  - spec URL input
 *
 * When a preset is selected, it pre-fills the spec URL and OAuth 2.0 fields
 * directly on the Ant Design form instance, then notifies the parent so it
 * can keep its own formValues state in sync (since setFieldsValue bypasses
 * the Form onValuesChange callback).
 */
const OpenAPIFormSection: React.FC<OpenAPIFormSectionProps> = ({
  form,
  accessToken,
  onValuesChange,
}) => {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);

  const handlePresetSelect = (entry: OpenAPIRegistryEntry) => {
    setSelectedPreset(entry.name);
    const updates = {
      spec_path: entry.spec_url,
      auth_type: AUTH_TYPE.OAUTH2,
      credentials: {
        authorization_url: entry.oauth.authorization_url,
        token_url: entry.oauth.token_url,
      },
    };
    form.setFieldsValue(updates);
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
        />
      </Form.Item>
    </>
  );
};

export default OpenAPIFormSection;
