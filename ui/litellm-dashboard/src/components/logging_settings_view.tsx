import React from "react";
import { Tag } from "antd";
import { CogIcon, BanIcon } from "@heroicons/react/outline";
import { callbackInfo, callback_map, reverse_callback_map } from "./callback_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";

interface LoggingConfig {
  callback_name: string;
  callback_type: string;
  callback_vars: Record<string, string>;
}

interface LoggingSettingsViewProps {
  loggingConfigs?: LoggingConfig[];
  disabledCallbacks?: string[];
  // Destinations this identity assigned itself, via metadata.logging_exporters.
  loggingExporters?: string[];
  // Destinations that target this identity via the credential's own scope
  // (credential_info.access.{teams,orgs,global}) -- the other direction. The
  // resolver unions both at request time; the UI unions them here for display.
  scopedExporters?: string[];
  variant?: "card" | "inline";
  className?: string;
}

export function LoggingSettingsView({
  loggingConfigs = [],
  disabledCallbacks = [],
  loggingExporters = [],
  scopedExporters = [],
  variant = "card",
  className = "",
}: LoggingSettingsViewProps) {
  const getLoggingDisplayName = (callbackName: string) => {
    // Find the display name for the callback
    const callbackDisplayName = Object.entries(callback_map).find(([_, value]) => value === callbackName)?.[0];
    return callbackDisplayName || callbackName;
  };

  const getEventTypeColor = (eventType: string): string | undefined => {
    switch (eventType) {
      case "success":
        return "green";
      case "failure":
        return "red";
      case "success_and_failure":
        return "blue";
      default:
        return undefined;
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
      {/* Logging Exporters: the union of destinations routing to this identity.
          Own = destinations this identity listed in its metadata.logging_exporters.
          Via scope = destinations whose credential_info.access targets this identity
          (a team/org id, or global). Both directions count; we render them together,
          marking how each entry was resolved. */}
      <div className="space-y-3">
        {(() => {
          const ownSet = new Set(loggingExporters);
          const scopedOnly = scopedExporters.filter((name) => !ownSet.has(name));
          const entries = [
            ...loggingExporters.map((name) => ({ name, source: "own" as const })),
            ...scopedOnly.map((name) => ({ name, source: "scope" as const })),
          ];
          return (
            <>
              <div className="flex items-center gap-2">
                <CogIcon className="h-4 w-4 text-blue-600" />
                <span className="font-semibold text-gray-900">Logging Exporters</span>
                <Tag color="blue">{entries.length}</Tag>
              </div>
              {entries.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {entries.map((entry, index) => (
                    <Tag key={index} color={entry.source === "own" ? "blue" : "geekblue"}>
                      {entry.name}
                      {entry.source === "scope" ? " (via scope)" : ""}
                    </Tag>
                  ))}
                </div>
              ) : (
                <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
                  <CogIcon className="h-4 w-4 text-gray-400" />
                  <span className="text-gray-500 text-sm">No logging exporters assigned</span>
                </div>
              )}
            </>
          );
        })()}
      </div>

      {/* Logging Integrations Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <CogIcon className="h-4 w-4 text-blue-600" />
          <span className="font-semibold text-gray-900">Logging Integrations</span>
          <Tag color="blue">{loggingConfigs.length}</Tag>
        </div>

        {loggingConfigs.length > 0 ? (
          <div className="space-y-3">
            {loggingConfigs.map((config, index) => {
              const displayName = getLoggingDisplayName(config.callback_name);

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-blue-50 border border-blue-200"
                >
                  <div className="flex items-center gap-3">
                    <Logo
                      src={callbackInfo[displayName]?.logo}
                      label={displayName}
                      className="w-5 h-5 object-contain"
                    />
                    <div>
                      <span className="block font-medium text-blue-800">{displayName}</span>
                      <span className="block text-xs text-blue-600">
                        {Object.keys(config.callback_vars).length} parameters configured
                      </span>
                    </div>
                  </div>
                  <Tag color={getEventTypeColor(config.callback_type)}>{getEventTypeLabel(config.callback_type)}</Tag>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
            <CogIcon className="h-4 w-4 text-gray-400" />
            <span className="text-gray-500 text-sm">No logging integrations configured</span>
          </div>
        )}
      </div>

      {/* Disabled Callbacks Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <BanIcon className="h-4 w-4 text-red-600" />
          <span className="font-semibold text-gray-900">Disabled Callbacks</span>
          <Tag color="red">{disabledCallbacks.length}</Tag>
        </div>

        {disabledCallbacks.length > 0 ? (
          <div className="space-y-3">
            {disabledCallbacks.map((callbackName, index) => {
              // Handle both display names and internal values
              const displayName = reverse_callback_map[callbackName] || callbackName;

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-red-50 border border-red-200"
                >
                  <div className="flex items-center gap-3">
                    <Logo
                      src={callbackInfo[displayName]?.logo}
                      label={displayName}
                      className="w-5 h-5 object-contain"
                    />
                    <div>
                      <span className="block font-medium text-red-800">{displayName}</span>
                      <span className="block text-xs text-red-600">Disabled for this key</span>
                    </div>
                  </div>
                  <Tag color="red">Disabled</Tag>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
            <BanIcon className="h-4 w-4 text-gray-400" />
            <span className="text-gray-500 text-sm">No callbacks disabled</span>
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
            <span className="block font-semibold text-gray-900">Logging Settings</span>
            <span className="block text-xs text-gray-500">
              Active logging integrations and disabled callbacks for this key
            </span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <span className="block font-medium text-gray-900 mb-3">Logging Settings</span>
      {content}
    </div>
  );
}

export default LoggingSettingsView;
