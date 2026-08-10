import React, { useState } from "react";
import { Select, Tooltip, Divider, Switch, Checkbox, Form } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";

const { Option } = Select;

interface KeyLifecycleSettingsProps {
  form: any; // Form instance from parent
  autoRotationEnabled: boolean;
  onAutoRotationChange: (enabled: boolean) => void;
  rotationInterval: string;
  onRotationIntervalChange: (interval: string) => void;
  isCreateMode?: boolean; // If true, shows "leave empty to never expire" instead of "-1 to never expire"
  neverExpire?: boolean;
  onNeverExpireChange?: (checked: boolean) => void;
  labels?: {
    expirySettings: string;
    expireKey: string;
    expiryTooltip: string;
    neverExpire: string;
    createPlaceholder: string;
    editPlaceholder: string;
    rotationSettings: string;
    enableRotation: string;
    rotationTooltip: string;
    rotationInterval: string;
    rotationIntervalTooltip: string;
    selectInterval: string;
    days: string;
    customInterval: string;
    customPlaceholder: string;
    supportedFormats: string;
    rotationNotice: string;
  };
}

const KeyLifecycleSettings: React.FC<KeyLifecycleSettingsProps> = ({
  form,
  autoRotationEnabled,
  onAutoRotationChange,
  rotationInterval,
  onRotationIntervalChange,
  isCreateMode = false,
  neverExpire = false,
  onNeverExpireChange,
  labels,
}) => {
  const text = {
    expirySettings: labels?.expirySettings ?? "Key Expiry Settings",
    expireKey: labels?.expireKey ?? "Expire Key",
    expiryTooltip:
      labels?.expiryTooltip ??
      "Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days). Leave empty to keep the current expiry unchanged.",
    neverExpire: labels?.neverExpire ?? "Never Expire",
    createPlaceholder: labels?.createPlaceholder ?? "e.g., 30d or leave empty to never expire",
    editPlaceholder: labels?.editPlaceholder ?? "e.g., 30d",
    rotationSettings: labels?.rotationSettings ?? "Auto-Rotation Settings",
    enableRotation: labels?.enableRotation ?? "Enable Auto-Rotation",
    rotationTooltip:
      labels?.rotationTooltip ?? "Key will automatically regenerate at the specified interval for enhanced security.",
    rotationInterval: labels?.rotationInterval ?? "Rotation Interval",
    rotationIntervalTooltip:
      labels?.rotationIntervalTooltip ??
      "How often the key should be automatically rotated. Choose the interval that best fits your security requirements.",
    selectInterval: labels?.selectInterval ?? "Select interval",
    days: labels?.days ?? "days",
    customInterval: labels?.customInterval ?? "Custom interval",
    customPlaceholder: labels?.customPlaceholder ?? "e.g., 1s, 5m, 2h, 14d",
    supportedFormats: labels?.supportedFormats ?? "Supported formats: seconds (s), minutes (m), hours (h), days (d)",
    rotationNotice:
      labels?.rotationNotice ??
      "When rotation occurs, you'll receive a notification with the new key. The old key will be deactivated after a brief grace period.",
  };
  // Predefined intervals
  const predefinedIntervals = ["7d", "30d", "90d", "180d", "365d"];

  // Check if current interval is custom
  const isCustomInterval = rotationInterval && !predefinedIntervals.includes(rotationInterval);

  const [showCustomInput, setShowCustomInput] = useState(isCustomInterval);
  const [customInterval, setCustomInterval] = useState(isCustomInterval ? rotationInterval : "");

  const handleIntervalChange = (value: string) => {
    if (value === "custom") {
      setShowCustomInput(true);
      // Don't change the actual interval yet, wait for custom input
    } else {
      setShowCustomInput(false);
      setCustomInterval("");
      onRotationIntervalChange(value);
    }
  };

  const handleCustomIntervalChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    setCustomInterval(value);
    onRotationIntervalChange(value);
  };

  return (
    <div className="space-y-6">
      {/* Key Expiry Section */}
      <div className="space-y-4">
        <span className="text-sm font-medium text-gray-700">{text.expirySettings}</span>

        <div className="space-y-2">
          <label className="text-sm font-medium text-gray-700 flex items-center space-x-1">
            <span>{text.expireKey}</span>
            <Tooltip title={text.expiryTooltip}>
              <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
            </Tooltip>
            {!isCreateMode && onNeverExpireChange && (
              <Checkbox
                checked={neverExpire}
                onChange={(e) => {
                  const checked = e.target.checked;
                  onNeverExpireChange(checked);
                  if (checked) {
                    if (form && typeof form.setFieldValue === "function") {
                      form.setFieldValue("duration", "");
                    } else if (form && typeof form.setFieldsValue === "function") {
                      form.setFieldsValue({ duration: "" });
                    }
                  }
                }}
                className="ml-2 text-sm font-normal text-gray-600"
              >
                {text.neverExpire}
              </Checkbox>
            )}
          </label>
          <Form.Item name="duration" noStyle initialValue="">
            <TextInput
              placeholder={isCreateMode ? text.createPlaceholder : text.editPlaceholder}
              className="w-full"
              disabled={!isCreateMode && neverExpire}
            />
          </Form.Item>
        </div>
      </div>

      <Divider />

      {/* Auto-Rotation Section */}
      <div className="space-y-4">
        <span className="text-sm font-medium text-gray-700">{text.rotationSettings}</span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-700 flex items-center space-x-1">
              <span>{text.enableRotation}</span>
              <Tooltip title={text.rotationTooltip}>
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
                <span>{text.rotationInterval}</span>
                <Tooltip title={text.rotationIntervalTooltip}>
                  <InfoCircleOutlined className="text-gray-400 cursor-help text-xs" />
                </Tooltip>
              </label>
              <div className="space-y-2">
                <Select
                  value={showCustomInput ? "custom" : rotationInterval}
                  onChange={handleIntervalChange}
                  className="w-full"
                  placeholder={text.selectInterval}
                >
                  <Option value="7d">7 {text.days}</Option>
                  <Option value="30d">30 {text.days}</Option>
                  <Option value="90d">90 {text.days}</Option>
                  <Option value="180d">180 {text.days}</Option>
                  <Option value="365d">365 {text.days}</Option>
                  <Option value="custom">{text.customInterval}</Option>
                </Select>

                {showCustomInput && (
                  <div className="space-y-1">
                    <TextInput
                      value={customInterval}
                      onChange={handleCustomIntervalChange}
                      placeholder={text.customPlaceholder}
                    />
                    <div className="text-xs text-gray-500">{text.supportedFormats}</div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {autoRotationEnabled && (
          <div className="bg-blue-50 p-3 rounded-md text-sm text-blue-700">{text.rotationNotice}</div>
        )}
      </div>
    </div>
  );
};

export default KeyLifecycleSettings;
