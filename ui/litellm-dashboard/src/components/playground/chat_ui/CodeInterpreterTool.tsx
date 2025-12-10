import React from "react";
import { Switch, Tooltip } from "antd";
import { CodeOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";

interface CodeInterpreterToolProps {
  accessToken: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  selectedContainerId: string | null;
  onContainerChange: (containerId: string | null) => void;
  disabled?: boolean;
}

const CodeInterpreterTool: React.FC<CodeInterpreterToolProps> = ({
  enabled,
  onEnabledChange,
  disabled = false,
}) => {
  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-gradient-to-r from-blue-50 to-purple-50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CodeOutlined className="text-blue-500" />
          <Text className="font-medium text-gray-700">Code Interpreter</Text>
          <Tooltip title="Run Python code to generate files, charts, and analyze data. Container is created automatically.">
            <InfoCircleOutlined className="text-gray-400 text-xs" />
          </Tooltip>
        </div>
        <Switch
          checked={enabled}
          onChange={onEnabledChange}
          disabled={disabled}
          size="small"
          className={enabled ? "bg-blue-500" : ""}
        />
      </div>
    </div>
  );
};

export default CodeInterpreterTool;
