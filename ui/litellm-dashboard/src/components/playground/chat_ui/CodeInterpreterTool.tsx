import React from "react";
import { Switch, Tooltip, message } from "antd";
import { CodeOutlined, InfoCircleOutlined, ExclamationCircleOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";

interface CodeInterpreterToolProps {
  accessToken: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  selectedContainerId: string | null;
  onContainerChange: (containerId: string | null) => void;
  selectedModel: string;
  disabled?: boolean;
}

const GITHUB_FEATURE_REQUEST_URL = "https://github.com/BerriAI/litellm/issues/new?template=feature_request.yml";

const isOpenAIModel = (model: string): boolean => {
  if (!model) return false;
  const lowerModel = model.toLowerCase();
  return (
    lowerModel.startsWith("openai/") ||
    lowerModel.startsWith("gpt-") ||
    lowerModel.startsWith("o1") ||
    lowerModel.startsWith("o3") ||
    lowerModel.includes("openai")
  );
};

const CodeInterpreterTool: React.FC<CodeInterpreterToolProps> = ({
  enabled,
  onEnabledChange,
  selectedModel,
  disabled = false,
}) => {
  const isOpenAI = isOpenAIModel(selectedModel);
  const isDisabled = disabled || !isOpenAI;

  const handleToggle = (checked: boolean) => {
    if (checked && !isOpenAI) {
      message.warning("Code Interpreter is only available for OpenAI models");
      return;
    }
    onEnabledChange(checked);
  };

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
          checked={enabled && isOpenAI}
          onChange={handleToggle}
          disabled={isDisabled}
          size="small"
          className={enabled && isOpenAI ? "bg-blue-500" : ""}
        />
      </div>
      
      {!isOpenAI && (
        <div className="mt-2 pt-2 border-t border-gray-200">
          <div className="flex items-start gap-2">
            <ExclamationCircleOutlined className="text-amber-500 mt-0.5" />
            <div className="text-xs text-gray-600">
              <span>Code Interpreter is currently only supported for OpenAI models. </span>
              <a
                href={GITHUB_FEATURE_REQUEST_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline"
              >
                Request support for other providers
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterTool;
