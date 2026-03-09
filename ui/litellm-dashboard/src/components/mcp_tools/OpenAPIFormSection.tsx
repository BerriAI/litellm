import React, { useState } from "react";
import { Form, Input, Tooltip } from "antd";
import { InfoCircleOutlined, ToolOutlined } from "@ant-design/icons";
import { FormInstance } from "antd/es/form";
import { AUTH_TYPE } from "./types";
import OpenAPIQuickPicker, { OpenAPIRegistryEntry, OpenAPIKeyTool } from "./OpenAPIQuickPicker";

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
 *  - curated key tools preview (8 tools from registry, expandable)
 */
const OpenAPIFormSection: React.FC<OpenAPIFormSectionProps> = ({
  form,
  accessToken,
  onValuesChange,
}) => {
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [keyTools, setKeyTools] = useState<OpenAPIKeyTool[]>([]);

  const handlePresetSelect = (entry: OpenAPIRegistryEntry) => {
    setSelectedPreset(entry.name);
    setKeyTools(entry.key_tools ?? []);
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

      {keyTools.length > 0 && (
        <KeyToolsPreview tools={keyTools} />
      )}
    </>
  );
};

interface KeyToolsPreviewProps {
  tools: OpenAPIKeyTool[];
}

const KeyToolsPreview: React.FC<KeyToolsPreviewProps> = ({ tools }) => {
  const [expanded, setExpanded] = useState(false);
  const shown = expanded ? tools : tools.slice(0, 4);

  return (
    <div className="mb-4 p-3 bg-blue-50 border border-blue-100 rounded-lg">
      <div className="flex items-center gap-1.5 mb-2">
        <ToolOutlined className="text-blue-500 text-xs" />
        <span className="text-xs font-medium text-blue-700">
          Key tools from this API
        </span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {shown.map((tool) => (
          <Tooltip key={tool.name} title={tool.description} placement="top">
            <span className="inline-flex items-center px-2 py-0.5 rounded bg-white border border-blue-200 text-xs text-blue-800 font-mono cursor-default hover:border-blue-400 transition-colors">
              {tool.name}
            </span>
          </Tooltip>
        ))}
      </div>

      {tools.length > 4 && (
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="mt-2 text-xs text-blue-600 hover:text-blue-800 cursor-pointer bg-transparent border-none p-0"
        >
          {expanded
            ? "Show less"
            : `+ ${tools.length - 4} more — all tools load from the spec below`}
        </button>
      )}
    </div>
  );
};

export default OpenAPIFormSection;
