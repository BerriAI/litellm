import React, { useState } from "react";
import { Card, Text } from "@tremor/react";
import { Input } from "antd";
import { SettingsIcon } from "lucide-react";
import ModelSelector from "../../common_components/ModelSelector";

interface ModelConfigCardProps {
  model: string;
  temperature?: number;
  maxTokens?: number;
  accessToken: string | null;
  onModelChange: (model: string) => void;
  onTemperatureChange: (temp: number) => void;
  onMaxTokensChange: (tokens: number) => void;
}

const ModelConfigCard: React.FC<ModelConfigCardProps> = ({
  model,
  temperature = 1,
  maxTokens = 1000,
  accessToken,
  onModelChange,
  onTemperatureChange,
  onMaxTokensChange,
}) => {
  const [showConfig, setShowConfig] = useState(false);

  return (
    <Card className="p-3">
      <div className="mb-3">
        <Text className="block mb-2 text-sm font-medium">Model</Text>
        <ModelSelector
          accessToken={accessToken || ""}
          value={model}
          onChange={onModelChange}
          showLabel={false}
        />
      </div>

      <button
        onClick={() => setShowConfig(!showConfig)}
        className="flex items-center text-xs font-medium text-gray-600 hover:text-gray-900"
      >
        <SettingsIcon size={14} className="mr-1" />
        <span>Configuration</span>
      </button>

      {showConfig && (
        <div className="mt-3 pt-3 border-t border-gray-200 -mx-3 px-0">
          <div className="space-y-3 px-3">
            <div>
              <div className="flex items-center justify-between mb-2">
                <Text className="text-xs text-gray-700">Temperature</Text>
                <Input
                  type="number"
                  size="small"
                  min={0}
                  max={2}
                  step={0.1}
                  value={temperature}
                  onChange={(e) => onTemperatureChange(parseFloat(e.target.value) || 0)}
                  className="w-16"
                />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <Text className="text-xs text-gray-700">Max Tokens</Text>
                <Input
                  type="number"
                  size="small"
                  min={1}
                  max={32768}
                  value={maxTokens}
                  onChange={(e) => onMaxTokensChange(parseInt(e.target.value) || 1000)}
                  className="w-20"
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default ModelConfigCard;

