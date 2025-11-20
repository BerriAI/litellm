import React, { useState } from "react";
import { Text } from "@tremor/react";
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
    <div className="flex items-center gap-3">
      <div className="w-[300px]">
        <ModelSelector
          accessToken={accessToken || ""}
          value={model}
          onChange={onModelChange}
          showLabel={false}
        />
      </div>

      <button
        onClick={() => setShowConfig(!showConfig)}
        className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
      >
        <SettingsIcon size={16} />
        <span>Parameters</span>
      </button>

      {showConfig && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-30">
          <div className="bg-white rounded-lg shadow-xl p-6 w-96">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Model Parameters</h3>
              <button
                onClick={() => setShowConfig(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                ✕
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Text className="text-sm text-gray-700">Temperature</Text>
                  <Input
                    type="number"
                    size="small"
                    min={0}
                    max={2}
                    step={0.1}
                    value={temperature}
                    onChange={(e) => onTemperatureChange(parseFloat(e.target.value) || 0)}
                    className="w-20"
                  />
                </div>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <Text className="text-sm text-gray-700">Max Tokens</Text>
                  <Input
                    type="number"
                    size="small"
                    min={1}
                    max={32768}
                    value={maxTokens}
                    onChange={(e) => onMaxTokensChange(parseInt(e.target.value) || 1000)}
                    className="w-24"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelConfigCard;

