import { InfoCircleOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";
import { Checkbox, InputNumber, Slider, Tooltip } from "antd";
import React, { useEffect, useState } from "react";

interface AdditionalModelSettingsProps {
  temperature?: number;
  maxTokens?: number;
  useAdvancedParams?: boolean;
  onTemperatureChange?: (value: number) => void;
  onMaxTokensChange?: (value: number) => void;
  onUseAdvancedParamsChange?: (value: boolean) => void;
}

const AdditionalModelSettings: React.FC<AdditionalModelSettingsProps> = ({
  temperature = 1.0,
  maxTokens = 2048,
  useAdvancedParams: externalUseAdvancedParams,
  onTemperatureChange,
  onMaxTokensChange,
  onUseAdvancedParamsChange,
}) => {
  const [internalUseAdvancedParams, setInternalUseAdvancedParams] = useState(false);
  const useAdvancedParams =
    externalUseAdvancedParams !== undefined ? externalUseAdvancedParams : internalUseAdvancedParams;
  const [localTemperature, setLocalTemperature] = useState(temperature);
  const [localMaxTokens, setLocalMaxTokens] = useState(maxTokens);

  // Sync local state with props when they change
  useEffect(() => {
    setLocalTemperature(temperature);
  }, [temperature]);

  useEffect(() => {
    setLocalMaxTokens(maxTokens);
  }, [maxTokens]);

  const handleTemperatureChange = (value: number | null) => {
    const newValue = value ?? 1.0;
    setLocalTemperature(newValue);
    onTemperatureChange?.(newValue);
  };

  const handleMaxTokensChange = (value: number | null) => {
    const newValue = value ?? 1000;
    setLocalMaxTokens(newValue);
    onMaxTokensChange?.(newValue);
  };

  const disabledOpacity = useAdvancedParams ? 1 : 0.4;
  const disabledTextColor = useAdvancedParams ? "text-gray-700" : "text-gray-400";

  const handleUseAdvancedParamsChange = (checked: boolean) => {
    if (onUseAdvancedParamsChange) {
      onUseAdvancedParamsChange(checked);
    } else {
      setInternalUseAdvancedParams(checked);
    }
  };

  return (
    <div className="space-y-4 p-4 w-80">
      <Checkbox checked={useAdvancedParams} onChange={(e) => handleUseAdvancedParamsChange(e.target.checked)}>
        <span className="font-medium">Use Advanced Parameters</span>
      </Checkbox>

      <div className="space-y-4 transition-opacity duration-200" style={{ opacity: disabledOpacity }}>
        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1">
              <Text className={`text-sm ${disabledTextColor}`}>Temperature</Text>
              <Tooltip title="Controls randomness. Lower values make output more deterministic, higher values more creative.">
                <InfoCircleOutlined className={`text-xs ${disabledTextColor} cursor-help`} />
              </Tooltip>
            </div>
            <InputNumber
              min={0}
              max={2}
              step={0.1}
              value={localTemperature}
              onChange={handleTemperatureChange}
              disabled={!useAdvancedParams}
              precision={1}
              className="w-20"
            />
          </div>
          <Slider
            min={0}
            max={2}
            step={0.1}
            value={localTemperature}
            onChange={handleTemperatureChange}
            disabled={!useAdvancedParams}
            marks={{
              0: "0",
              1: "1.0",
              2: "2.0",
            }}
          />
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1">
              <Text className={`text-sm ${disabledTextColor}`}>Max Tokens</Text>
              <Tooltip title="Maximum number of tokens to generate in the response.">
                <InfoCircleOutlined className={`text-xs ${disabledTextColor} cursor-help`} />
              </Tooltip>
            </div>
            <InputNumber
              min={1}
              max={32768}
              step={1}
              value={localMaxTokens}
              onChange={handleMaxTokensChange}
              disabled={!useAdvancedParams}
            />
          </div>
          <Slider
            min={1}
            max={32768}
            step={1}
            value={localMaxTokens}
            onChange={handleMaxTokensChange}
            disabled={!useAdvancedParams}
            marks={{
              1: "1",
              32768: "32768",
            }}
          />
        </div>
      </div>
    </div>
  );
};

export default AdditionalModelSettings;
