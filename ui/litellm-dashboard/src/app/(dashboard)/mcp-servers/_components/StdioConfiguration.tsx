import React from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { MountedFormField, bindControl } from "@/components/common_components/MountedFormField";

interface StdioConfigurationProps {
  isVisible: boolean;
  /**
   * When true, stdio_config is required + validated as JSON.
   * Edit screen can set this to false when using dedicated command/args/env fields.
   */
  required?: boolean;
}

const StdioConfiguration: React.FC<StdioConfigurationProps> = ({ isVisible, required = true }) => {
  if (!isVisible) return null;

  return (
    <MountedFormField
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          Stdio Configuration (JSON)
          <Tooltip title="Paste your stdio MCP server configuration in JSON format. You can use the full mcpServers structure from config.yaml or just the inner server configuration.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name="stdio_config"
      required={required}
      rules={{
        validate: (value) => {
          if (!value) return required ? "Please enter stdio configuration" : true;
          try {
            JSON.parse(String(value));
            return true;
          } catch {
            return "Please enter valid JSON";
          }
        },
      }}
    >
      {(field) => (
        <Input.TextArea
          {...bindControl<string | undefined>(field)}
          placeholder={`{
  "mcpServers": {
    "circleci-mcp-server": {
      "command": "npx",
      "args": ["-y", "@circleci/mcp-server-circleci"],
      "env": {
        "CIRCLECI_TOKEN": "your-circleci-token",
        "CIRCLECI_BASE_URL": "https://circleci.com"
      }
    }
  }
}`}
          rows={12}
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 font-mono text-sm"
        />
      )}
    </MountedFormField>
  );
};

export default StdioConfiguration;
