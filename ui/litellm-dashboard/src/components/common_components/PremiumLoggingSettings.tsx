import React from "react";
import LoggingSettings from "../team/LoggingSettings";

interface PremiumLoggingSettingsProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  value: any[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChange: (settings: any[]) => void;
  premiumUser?: boolean;
  disabledCallbacks?: string[];
  onDisabledCallbacksChange?: (disabledCallbacks: string[]) => void;
}

export function PremiumLoggingSettings({
  value,
  onChange,
  premiumUser = false,
  disabledCallbacks = [],
  onDisabledCallbacksChange,
}: PremiumLoggingSettingsProps) {
  if (!premiumUser) {
    return (
      <div>
        <div className="flex flex-wrap gap-2 mb-3">
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900 text-emerald-800 dark:text-emerald-200 text-sm font-medium opacity-50">
            ✨ langfuse-logging
          </div>
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900 text-emerald-800 dark:text-emerald-200 text-sm font-medium opacity-50">
            ✨ datadog-logging
          </div>
        </div>
        <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg">
          <p className="text-sm text-amber-800 dark:text-amber-200">
            Setting Key/Team logging settings is a LiteLLM Enterprise feature.
            Global Logging Settings are available for all free users. Get a
            trial key{" "}
            <a
              href="https://www.litellm.ai/#pricing"
              target="_blank"
              rel="noopener noreferrer"
              className="underline"
            >
              here
            </a>
            .
          </p>
        </div>
      </div>
    );
  }

  return (
    <LoggingSettings
      value={value}
      onChange={onChange}
      disabledCallbacks={disabledCallbacks}
      onDisabledCallbacksChange={onDisabledCallbacksChange}
    />
  );
}

export default PremiumLoggingSettings;
