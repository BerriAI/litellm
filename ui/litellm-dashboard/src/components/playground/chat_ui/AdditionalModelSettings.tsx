import { InfoCircleOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";
import { Checkbox, InputNumber, Popover, Slider, Tooltip, Typography } from "antd";
import React, { useEffect, useState } from "react";

interface AdditionalModelSettingsProps {
  temperature?: number;
  maxTokens?: number;
  useAdvancedParams?: boolean;
  onTemperatureChange?: (value: number) => void;
  onMaxTokensChange?: (value: number) => void;
  onUseAdvancedParamsChange?: (value: boolean) => void;
  mockTestFallbacks?: boolean;
  onMockTestFallbacksChange?: (value: boolean) => void;
}

const AdditionalModelSettings: React.FC<AdditionalModelSettingsProps> = ({
  temperature = 1.0,
  maxTokens = 2048,
  useAdvancedParams: externalUseAdvancedParams,
  onTemperatureChange,
  onMaxTokensChange,
  onUseAdvancedParamsChange,
  mockTestFallbacks,
  onMockTestFallbacksChange,
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

      {onMockTestFallbacksChange && (
        <div className="flex items-center gap-1">
          <Checkbox
            checked={mockTestFallbacks ?? false}
            onChange={(e) => onMockTestFallbacksChange(e.target.checked)}
          >
            <span className="font-medium">Simulate failure to test fallbacks</span>
          </Checkbox>
          <Popover
            trigger="hover"
            placement="right"
            content={
              <div style={{ maxWidth: 340 }}>
                <Typography.Paragraph className="text-sm" style={{ marginBottom: 8 }}>
                  Causes the first request to fail so the router tries fallbacks (if configured). Use
                  this to verify your fallback setup.
                </Typography.Paragraph>
                <Typography.Paragraph className="text-sm" style={{ marginBottom: 0 }}>
                  Behavior can differ when keys, teams, or router settings are configured.{" "}
                  <a
                    href="https://docs.litellm.ai/docs/proxy/keys_teams_router_settings"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800"
                  >
                    Learn more
                  </a>
                </Typography.Paragraph>
              </div>
            }
          >
            <InfoCircleOutlined
              className="text-xs text-gray-400 cursor-pointer shrink-0 hover:text-gray-600"
              aria-label="Help: Simulate failure to test fallbacks"
            />
          </Popover>
        </div>
      )}

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
