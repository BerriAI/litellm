import React from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { antdRequired } from "@/components/common_components/antdFormRules";
import { parsesAsJson, textControl } from "./mcpFieldRules";

interface StdioConfigurationProps {
  isVisible: boolean;
  /**
   * When true, stdio_config is required + validated as JSON.
   * Edit screen can set this to false when using dedicated command/args/env fields.
   */
  required?: boolean;
}

const PLACEHOLDER = `{
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
}`;

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
        validate: {
          ...(required ? { required: antdRequired("Please enter stdio configuration") } : {}),
          json: parsesAsJson("Please enter valid JSON"),
        },
      }}
    >
      {(control) => (
        <Input.TextArea
          {...textControl(control)}
          placeholder={PLACEHOLDER}
          rows={12}
          className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 font-mono text-sm"
        />
      )}
    </MountedFormField>
  );
};

export default StdioConfiguration;
