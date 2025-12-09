import React, { useState } from "react";
import { Switch, Tooltip, message } from "antd";
import {
  CodeOutlined,
  InfoCircleOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import { Text } from "@tremor/react";
import { getProxyBaseUrl } from "@/components/networking";

interface CodeInterpreterToolProps {
  accessToken: string;
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  selectedContainerId: string | null;
  onContainerChange: (containerId: string | null) => void;
  disabled?: boolean;
}

const SAMPLE_PROMPTS = [
  "Generate a CSV with sample sales data and create a chart",
  "Write a Python script to analyze stock prices",
  "Create a visualization of monthly trends",
  "Generate sample data and calculate statistics",
];

const CodeInterpreterTool: React.FC<CodeInterpreterToolProps> = ({
  accessToken,
  enabled,
  onEnabledChange,
  selectedContainerId,
  onContainerChange,
  disabled = false,
}) => {
  const [isCreatingContainer, setIsCreatingContainer] = useState(false);
  const proxyBaseUrl = getProxyBaseUrl();

  // Auto-create container when enabled
  const handleToggle = async (checked: boolean) => {
    if (checked && !selectedContainerId) {
      // Create a container automatically
      setIsCreatingContainer(true);
      try {
        const response = await fetch(`${proxyBaseUrl}/v1/containers`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            name: `code-interpreter-${Date.now()}`,
            expires_after: {
              anchor: "last_active_at",
              minutes: 20,
            },
          }),
        });

        if (!response.ok) {
          throw new Error("Failed to create container");
        }

        const data = await response.json();
        onContainerChange(data.id);
        onEnabledChange(true);
        message.success("Code Interpreter ready!");
      } catch (error) {
        console.error("Error creating container:", error);
        message.error("Failed to enable Code Interpreter");
        onEnabledChange(false);
      } finally {
        setIsCreatingContainer(false);
      }
    } else if (!checked) {
      onEnabledChange(false);
      // Keep container ID for potential reuse
    } else {
      onEnabledChange(checked);
    }
  };

  return (
    <div className="border border-gray-200 rounded-lg p-3 bg-gradient-to-r from-blue-50 to-purple-50">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CodeOutlined className="text-blue-500" />
          <Text className="font-medium text-gray-700">Code Interpreter</Text>
          <Tooltip title="Run Python code to generate files, charts, and analyze data">
            <InfoCircleOutlined className="text-gray-400 text-xs" />
          </Tooltip>
        </div>
        <div className="flex items-center gap-2">
          {isCreatingContainer && (
            <LoadingOutlined className="text-blue-500" spin />
          )}
          {enabled && selectedContainerId && !isCreatingContainer && (
            <CheckCircleOutlined className="text-green-500" />
          )}
          <Switch
            checked={enabled}
            onChange={handleToggle}
            disabled={disabled || isCreatingContainer}
            size="small"
            className={enabled ? "bg-blue-500" : ""}
          />
        </div>
      </div>

      {/* Sample prompts when enabled */}
      {enabled && selectedContainerId && (
        <div className="mt-3 pt-3 border-t border-gray-200">
          <Text className="text-xs text-gray-500 mb-2 block">Try asking:</Text>
          <div className="flex flex-wrap gap-1.5">
            {SAMPLE_PROMPTS.map((prompt, index) => (
              <button
                key={index}
                className="text-xs px-2 py-1 bg-white border border-gray-200 rounded-full hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 transition-colors truncate max-w-full"
                onClick={() => {
                  // Copy prompt to clipboard or trigger input
                  navigator.clipboard.writeText(prompt);
                  message.success("Copied to input!");
                }}
                title={prompt}
              >
                {prompt.length > 40 ? prompt.substring(0, 40) + "..." : prompt}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CodeInterpreterTool;
