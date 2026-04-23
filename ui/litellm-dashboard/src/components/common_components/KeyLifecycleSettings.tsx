import React, { useState } from "react";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface KeyLifecycleSettingsProps {
  // Form instance from parent (antd Form)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  form: any;
  autoRotationEnabled: boolean;
  onAutoRotationChange: (enabled: boolean) => void;
  rotationInterval: string;
  onRotationIntervalChange: (interval: string) => void;
  isCreateMode?: boolean;
  neverExpire?: boolean;
  onNeverExpireChange?: (checked: boolean) => void;
}

const InfoTooltip: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="h-3 w-3 text-muted-foreground cursor-help" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const KeyLifecycleSettings: React.FC<KeyLifecycleSettingsProps> = ({
  form,
  autoRotationEnabled,
  onAutoRotationChange,
  rotationInterval,
  onRotationIntervalChange,
  isCreateMode = false,
  neverExpire = false,
  onNeverExpireChange,
}) => {
  const predefinedIntervals = ["7d", "30d", "90d", "180d", "365d"];

  const isCustomInterval =
    rotationInterval && !predefinedIntervals.includes(rotationInterval);

  const [showCustomInput, setShowCustomInput] = useState(isCustomInterval);
  const [customInterval, setCustomInterval] = useState(
    isCustomInterval ? rotationInterval : "",
  );
  const [durationValue, setDurationValue] = useState<string>(
    form?.getFieldValue?.("duration") || "",
  );

  const handleIntervalChange = (value: string) => {
    if (value === "custom") {
      setShowCustomInput(true);
    } else {
      setShowCustomInput(false);
      setCustomInterval("");
      onRotationIntervalChange(value);
    }
  };

  const handleCustomIntervalChange = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const value = e.target.value;
    setCustomInterval(value);
    onRotationIntervalChange(value);
  };

  const handleDurationChange = (value: string) => {
    setDurationValue(value);
    if (form && typeof form.setFieldValue === "function") {
      form.setFieldValue("duration", value);
    } else if (form && typeof form.setFieldsValue === "function") {
      form.setFieldsValue({ duration: value });
    }
  };

  return (
    <div className="space-y-6">
      <div className="space-y-4">
        <span className="text-sm font-medium text-foreground">
          Key Expiry Settings
        </span>

        <div className="space-y-2">
          <div className="text-sm font-medium text-foreground flex items-center space-x-1">
            <span>Expire Key</span>
            <InfoTooltip>
              Set when this key should expire. Format: 30s (seconds), 30m
              (minutes), 30h (hours), 30d (days). Leave empty to keep the
              current expiry unchanged.
            </InfoTooltip>
            {!isCreateMode && onNeverExpireChange && (
              <label className="ml-2 inline-flex items-center gap-1 text-sm font-normal text-muted-foreground">
                <Checkbox
                  checked={neverExpire}
                  onCheckedChange={(c) => {
                    const checked = c === true;
                    onNeverExpireChange(checked);
                    if (checked) {
                      setDurationValue("");
                      if (form && typeof form.setFieldValue === "function") {
                        form.setFieldValue("duration", "");
                      } else if (
                        form &&
                        typeof form.setFieldsValue === "function"
                      ) {
                        form.setFieldsValue({ duration: "" });
                      }
                    }
                  }}
                />
                Never Expire
              </label>
            )}
          </div>
          <Input
            name="duration"
            placeholder={
              isCreateMode
                ? "e.g., 30d or leave empty to never expire"
                : "e.g., 30d"
            }
            className="w-full"
            value={durationValue}
            onChange={(e) => handleDurationChange(e.target.value)}
            disabled={!isCreateMode && neverExpire}
          />
        </div>
      </div>

      <hr className="border-border" />

      <div className="space-y-4">
        <span className="text-sm font-medium text-foreground">
          Auto-Rotation Settings
        </span>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground flex items-center space-x-1">
              <span>Enable Auto-Rotation</span>
              <InfoTooltip>
                Key will automatically regenerate at the specified interval
                for enhanced security.
              </InfoTooltip>
            </label>
            <Switch
              checked={autoRotationEnabled}
              onCheckedChange={onAutoRotationChange}
            />
          </div>

          {autoRotationEnabled && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground flex items-center space-x-1">
                <span>Rotation Interval</span>
                <InfoTooltip>
                  How often the key should be automatically rotated. Choose
                  the interval that best fits your security requirements.
                </InfoTooltip>
              </label>
              <div className="space-y-2">
                <Select
                  value={showCustomInput ? "custom" : rotationInterval}
                  onValueChange={handleIntervalChange}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select interval" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="7d">7 days</SelectItem>
                    <SelectItem value="30d">30 days</SelectItem>
                    <SelectItem value="90d">90 days</SelectItem>
                    <SelectItem value="180d">180 days</SelectItem>
                    <SelectItem value="365d">365 days</SelectItem>
                    <SelectItem value="custom">Custom interval</SelectItem>
                  </SelectContent>
                </Select>

                {showCustomInput && (
                  <div className="space-y-1">
                    <Input
                      value={customInterval}
                      onChange={handleCustomIntervalChange}
                      placeholder="e.g., 1s, 5m, 2h, 14d"
                    />
                    <div className="text-xs text-muted-foreground">
                      Supported formats: seconds (s), minutes (m), hours (h),
                      days (d)
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {autoRotationEnabled && (
          <div className="bg-blue-50 dark:bg-blue-950/30 p-3 rounded-md text-sm text-blue-700 dark:text-blue-300">
            When rotation occurs, you&apos;ll receive a notification with the
            new key. The old key will be deactivated after a brief grace
            period.
          </div>
        )}
      </div>
    </div>
  );
};

export default KeyLifecycleSettings;
