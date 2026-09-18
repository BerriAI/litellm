import React, { useState } from "react";
import { CircleHelp } from "lucide-react";

import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

const PREDEFINED_INTERVALS = ["7d", "30d", "90d", "180d", "365d"] as const;

const INTERVAL_LABELS: Record<string, string> = {
  "7d": "7 days",
  "30d": "30 days",
  "90d": "90 days",
  "180d": "180 days",
  "365d": "365 days",
  custom: "Custom interval",
};

interface KeyLifecycleSettingsProps {
  value?: string;
  onChange?: (value: string) => void;
  autoRotationEnabled: boolean;
  onAutoRotationChange: (enabled: boolean) => void;
  rotationInterval: string;
  onRotationIntervalChange: (interval: string) => void;
  isCreateMode?: boolean;
  neverExpire?: boolean;
  onNeverExpireChange?: (checked: boolean) => void;
  id?: string;
}

const hintIcon = (hint: string): React.ReactNode => (
  <Tooltip>
    <TooltipTrigger
      render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />}
      aria-label={hint}
    />
    <TooltipContent>{hint}</TooltipContent>
  </Tooltip>
);

const KeyLifecycleSettings: React.FC<KeyLifecycleSettingsProps> = ({
  value,
  onChange,
  autoRotationEnabled,
  onAutoRotationChange,
  rotationInterval,
  onRotationIntervalChange,
  isCreateMode = false,
  neverExpire = false,
  onNeverExpireChange,
  id,
}) => {
  const isCustomInterval = Boolean(rotationInterval) && !PREDEFINED_INTERVALS.includes(rotationInterval as never);

  const [showCustomInput, setShowCustomInput] = useState(isCustomInterval);
  const [customInterval, setCustomInterval] = useState(isCustomInterval ? rotationInterval : "");

  const durationId = id ?? "key-lifecycle-duration";

  const handleIntervalChange = (next: string) => {
    if (next === "custom") {
      setShowCustomInput(true);
      return;
    }
    setShowCustomInput(false);
    setCustomInterval("");
    onRotationIntervalChange(next);
  };

  const handleCustomIntervalChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setCustomInterval(event.target.value);
    onRotationIntervalChange(event.target.value);
  };

  const handleNeverExpireChange = (checked: boolean) => {
    onNeverExpireChange?.(checked);
    if (checked) {
      onChange?.("");
    }
  };

  return (
    <TooltipProvider>
      <div className="space-y-6">
        <div className="space-y-4">
          <span className="text-sm font-medium text-foreground">Key Expiry Settings</span>

          <div className="space-y-2">
            <div className="flex items-center space-x-1 text-sm font-medium text-foreground">
              <label htmlFor={durationId}>Expire Key</label>
              {hintIcon(
                "Set when this key should expire. Format: 30s (seconds), 30m (minutes), 30h (hours), 30d (days). Leave empty to keep the current expiry unchanged.",
              )}
              {!isCreateMode && onNeverExpireChange && (
                <span className="ml-2 flex items-center gap-2 text-sm font-normal text-muted-foreground">
                  <Checkbox
                    id={`${durationId}-never-expire`}
                    checked={neverExpire}
                    onCheckedChange={handleNeverExpireChange}
                  />
                  <label htmlFor={`${durationId}-never-expire`} className="cursor-pointer">
                    Never Expire
                  </label>
                </span>
              )}
            </div>
            <Input
              id={durationId}
              value={value ?? ""}
              onChange={(event) => onChange?.(event.target.value)}
              placeholder={isCreateMode ? "e.g., 30d or leave empty to never expire" : "e.g., 30d"}
              className="w-full"
              disabled={!isCreateMode && neverExpire}
            />
          </div>
        </div>

        <Separator />

        <div className="space-y-4">
          <span className="text-sm font-medium text-foreground">Auto-Rotation Settings</span>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label className="flex items-center space-x-1 text-sm font-medium text-foreground">
                <span>Enable Auto-Rotation</span>
                {hintIcon("Key will automatically regenerate at the specified interval for enhanced security.")}
              </label>
              <Switch checked={autoRotationEnabled} onCheckedChange={onAutoRotationChange} />
            </div>

            {autoRotationEnabled && (
              <div className="space-y-2">
                <label className="flex items-center space-x-1 text-sm font-medium text-foreground">
                  <span>Rotation Interval</span>
                  {hintIcon(
                    "How often the key should be automatically rotated. Choose the interval that best fits your security requirements.",
                  )}
                </label>
                <div className="space-y-2">
                  <Select
                    value={showCustomInput ? "custom" : rotationInterval || null}
                    onValueChange={(next: string | null) => next !== null && handleIntervalChange(next)}
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select interval">
                        {(selected: string | null) =>
                          selected === null ? (
                            "Select interval"
                          ) : (
                            <span title={INTERVAL_LABELS[selected] ?? selected}>
                              {INTERVAL_LABELS[selected] ?? selected}
                            </span>
                          )
                        }
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {PREDEFINED_INTERVALS.map((interval) => (
                        <SelectItem key={interval} value={interval} title={INTERVAL_LABELS[interval]}>
                          {INTERVAL_LABELS[interval]}
                        </SelectItem>
                      ))}
                      <SelectItem value="custom" title={INTERVAL_LABELS.custom}>
                        {INTERVAL_LABELS.custom}
                      </SelectItem>
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
                        Supported formats: seconds (s), minutes (m), hours (h), days (d)
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {autoRotationEnabled && (
            <div className="rounded-md bg-info/10 p-3 text-sm text-info">
              When rotation occurs, you&apos;ll receive a notification with the new key. The old key will be deactivated
              after a brief grace period.
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  );
};

export default KeyLifecycleSettings;
