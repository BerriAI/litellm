import React from "react";
import { Select, Tooltip, Divider, Switch } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";

const { Option } = Select;

interface KeyLifecycleSettingsProps {
  form: any; // Form instance from parent
  autoRotationEnabled: boolean;
  onAutoRotationChange: (enabled: boolean) => void;
  rotationInterval: string;
  onRotationIntervalChange: (interval: string) => void;
}

const KeyLifecycleSettings: React.FC<KeyLifecycleSettingsProps> = ({
  form,
  autoRotationEnabled,
  onAutoRotationChange,
  rotationInterval,
  onRotationIntervalChange,
}) => {
  return (
    <div className="space-y-6">
      {/* Key Expiry Section */}
      <div className="space-y-4">
        <span className="text-sm font-medium text-gray-700">Key Expiry Settings</span>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 flex items-center space-x-1">
            <span>Expire Key</span>
            <Tooltip title="Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days)">
              <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
            </Tooltip>
          </label>
          <TextInput
            name="duration"
            placeholder="e.g., 30d"
            className="w-full"
          />
        </div>
      </div>

      <Divider />

      {/* Auto-Rotation Section */}
      <div className="space-y-4">
        <span className="text-sm font-medium text-gray-700">Auto-Rotation Settings</span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 flex items-center space-x-1">
              <span>Enable Auto-Rotation</span>
              <Tooltip title="Key will automatically regenerate at the specified interval for enhanced security.">
                <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
              </Tooltip>
            </label>
            <Switch
              checked={autoRotationEnabled}
              onChange={onAutoRotationChange}
              size="default"
              className={autoRotationEnabled ? "" : "bg-gray-400"}
            />
          </div>

          {autoRotationEnabled && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-gray-700 flex items-center space-x-1">
                <span>Rotation Interval</span>
                <Tooltip title="How often the key should be automatically rotated. Choose the interval that best fits your security requirements.">
                  <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
                </Tooltip>
              </label>
              <Select
                value={rotationInterval}
                onChange={onRotationIntervalChange}
                className="w-full"
                placeholder="Select interval"
              >
                <Option value="7d">7 days</Option>
                <Option value="30d">30 days</Option>
                <Option value="90d">90 days</Option>
                <Option value="180d">180 days</Option>
                <Option value="365d">365 days</Option>
              </Select>
            </div>
          )}
        </div>

        {autoRotationEnabled && (
          <div className="bg-blue-50 p-3 rounded-md text-sm text-blue-700">
            When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period.
          </div>
        )}
      </div>
    </div>
  );
};

export default KeyLifecycleSettings;
