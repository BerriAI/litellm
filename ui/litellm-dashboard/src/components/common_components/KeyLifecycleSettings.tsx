import React from "react";
import { Form, Switch, Select, Typography, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";

const { Text } = Typography;
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
    <div className="space-y-4">
      <Form.Item
        label={
          <span>
            Expire Key{' '}
            <Tooltip title="Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days)">
              <InfoCircleOutlined style={{ marginLeft: '4px' }} />
            </Tooltip>
          </span>
        }
        name="duration"
        className="mb-4"
      >
        <TextInput placeholder="e.g., 30d" />
      </Form.Item>

      <div className="mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Text className="text-sm font-medium text-gray-700">Enable Auto-Rotation</Text>
            <Tooltip title="Automatically rotate this API key at regular intervals for enhanced security. A new key will be generated and the old one will be deactivated.">
              <InfoCircleOutlined className="text-gray-400" style={{ fontSize: '14px' }} />
            </Tooltip>
          </div>
          <Switch
            checked={autoRotationEnabled}
            onChange={onAutoRotationChange}
            size="default"
          />
        </div>

        {autoRotationEnabled && (
          <div className="ml-4 pl-4 border-l-2 border-gray-200">
            <Form.Item
              label={
                <span>
                  Rotation Interval{' '}
                  <Tooltip title="How often the key should be automatically rotated. Choose the interval that best fits your security requirements.">
                    <InfoCircleOutlined style={{ marginLeft: '4px' }} />
                  </Tooltip>
                </span>
              }
              name="rotation_interval"
              className="mb-2"
            >
              <Select
                value={rotationInterval}
                onChange={onRotationIntervalChange}
                style={{ width: '200px' }}
                placeholder="Select interval"
              >
                <Option value="7d">7 days</Option>
                <Option value="30d">30 days</Option>
                <Option value="60d">60 days</Option>
                <Option value="90d">90 days</Option>
              </Select>
            </Form.Item>
            
            <Text className="text-xs text-gray-500 block">
              When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period.
            </Text>
          </div>
        )}
      </div>
    </div>
  );
};

export default KeyLifecycleSettings;
