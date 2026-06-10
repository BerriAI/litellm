import React from "react";
import { Text } from "@tremor/react";
import { Trans, useTranslation } from "react-i18next";
import LoggingSettings from "../team/LoggingSettings";

interface PremiumLoggingSettingsProps {
  value: any[];
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
  const { t } = useTranslation();

  if (!premiumUser) {
    return (
      <div>
        <div className="flex flex-wrap gap-2 mb-3">
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-green-50 border border-green-200 text-green-800 text-sm font-medium opacity-50">
            ✨ langfuse-logging
          </div>
          <div className="inline-flex items-center px-3 py-1.5 rounded-lg bg-green-50 border border-green-200 text-green-800 text-sm font-medium opacity-50">
            ✨ datadog-logging
          </div>
        </div>
        <div className="p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <Text className="text-sm text-yellow-800">
            <Trans
              i18nKey="commonComponents.premiumLoggingSettings.enterpriseNotice"
              components={{
                pricingLink: (
                  <a
                    href="https://www.litellm.ai/#pricing"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="underline"
                  />
                ),
              }}
            />
          </Text>
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
