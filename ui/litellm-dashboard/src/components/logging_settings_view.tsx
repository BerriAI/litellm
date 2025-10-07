import React from "react";
import { Text, Badge } from "@tremor/react";
import { CogIcon, BanIcon } from "@heroicons/react/outline";
import { callbackInfo, callback_map, reverse_callback_map } from "./callback_info_helpers";

interface LoggingConfig {
  callback_name: string;
  callback_type: string;
  callback_vars: Record<string, string>;
}

interface LoggingSettingsViewProps {
  loggingConfigs?: LoggingConfig[];
  disabledCallbacks?: string[];
  variant?: "card" | "inline";
  className?: string;
}

export function LoggingSettingsView({
  loggingConfigs = [],
  disabledCallbacks = [],
  variant = "card",
  className = "",
}: LoggingSettingsViewProps) {
  const getLoggingDisplayName = (callbackName: string) => {
    // Find the display name for the callback
    const callbackDisplayName = Object.entries(callback_map).find(([_, value]) => value === callbackName)?.[0];
    return callbackDisplayName || callbackName;
  };

  const getEventTypeColor = (eventType: string) => {
    switch (eventType) {
      case "success":
        return "green";
      case "failure":
        return "red";
      case "success_and_failure":
        return "blue";
      default:
        return "gray";
    }
  };

  const getEventTypeLabel = (eventType: string) => {
    switch (eventType) {
      case "success":
        return "Success Only";
      case "failure":
        return "Failure Only";
      case "success_and_failure":
        return "Success & Failure";
      default:
        return eventType;
    }
  };

  const content = (
    <div className="space-y-6">
      {/* Logging Integrations Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <CogIcon className="h-4 w-4 text-blue-600" />
          <Text className="font-semibold text-gray-900">Logging Integrations</Text>
          <Badge color="blue" size="xs">
            {loggingConfigs.length}
          </Badge>
        </div>

        {loggingConfigs.length > 0 ? (
          <div className="space-y-3">
            {loggingConfigs.map((config, index) => {
              const displayName = getLoggingDisplayName(config.callback_name);
              const logoUrl = callbackInfo[displayName]?.logo;

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-blue-50 border border-blue-200"
                >
                  <div className="flex items-center gap-3">
                    {logoUrl ? (
                      <img src={logoUrl} alt={displayName} className="w-5 h-5 object-contain" />
                    ) : (
                      <CogIcon className="h-5 w-5 text-gray-400" />
                    )}
                    <div>
                      <Text className="font-medium text-blue-800">{displayName}</Text>
                      <Text className="text-xs text-blue-600">
                        {Object.keys(config.callback_vars).length} parameters configured
                      </Text>
                    </div>
                  </div>
                  <Badge color={getEventTypeColor(config.callback_type)} size="sm">
                    {getEventTypeLabel(config.callback_type)}
                  </Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
            <CogIcon className="h-4 w-4 text-gray-400" />
            <Text className="text-gray-500 text-sm">No logging integrations configured</Text>
          </div>
        )}
      </div>

      {/* Disabled Callbacks Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <BanIcon className="h-4 w-4 text-red-600" />
          <Text className="font-semibold text-gray-900">Disabled Callbacks</Text>
          <Badge color="red" size="xs">
            {disabledCallbacks.length}
          </Badge>
        </div>

        {disabledCallbacks.length > 0 ? (
          <div className="space-y-3">
            {disabledCallbacks.map((callbackName, index) => {
              // Handle both display names and internal values
              const displayName = reverse_callback_map[callbackName] || callbackName;
              const logoUrl = callbackInfo[displayName]?.logo;

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-red-50 border border-red-200"
                >
                  <div className="flex items-center gap-3">
                    {logoUrl ? (
                      <img src={logoUrl} alt={displayName} className="w-5 h-5 object-contain" />
                    ) : (
                      <BanIcon className="h-5 w-5 text-gray-400" />
                    )}
                    <div>
                      <Text className="font-medium text-red-800">{displayName}</Text>
                      <Text className="text-xs text-red-600">Disabled for this key</Text>
                    </div>
                  </div>
                  <Badge color="red" size="sm">
                    Disabled
                  </Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
            <BanIcon className="h-4 w-4 text-gray-400" />
            <Text className="text-gray-500 text-sm">No callbacks disabled</Text>
          </div>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-white border border-gray-200 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <Text className="font-semibold text-gray-900">Logging Settings</Text>
            <Text className="text-xs text-gray-500">
              Active logging integrations and disabled callbacks for this key
            </Text>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <Text className="font-medium text-gray-900 mb-3">Logging Settings</Text>
      {content}
    </div>
  );
}

export default LoggingSettingsView;
