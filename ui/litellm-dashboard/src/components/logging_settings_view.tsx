import React from "react";
import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
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
  variant?: "card" | "inline";
  className?: string;
}

export function LoggingSettingsView({
  loggingConfigs = [],
  disabledCallbacks = [],
  variant = "card",
  className = "",
}: LoggingSettingsViewProps) {
  const { t } = useTranslation();
  const getLoggingDisplayName = (callbackName: string) => {
    // Find the display name for the callback
    const callbackDisplayName = Object.entries(callback_map).find(([_, value]) => value === callbackName)?.[0];
    return callbackDisplayName || callbackName;
  };

  const getEventTypeVariant = (eventType: string): React.ComponentProps<typeof Badge>["variant"] => {
    switch (eventType) {
      case "success":
        return "default";
      case "failure":
        return "destructive";
      case "success_and_failure":
        return "secondary";
      default:
        return "outline";
    }
  };

  const getEventTypeLabel = (eventType: string) => {
    switch (eventType) {
      case "success":
        return t("loggingSettingsView.eventTypes.success");
      case "failure":
        return t("loggingSettingsView.eventTypes.failure");
      case "success_and_failure":
        return t("loggingSettingsView.eventTypes.successAndFailure");
      default:
        return eventType;
    }
  };

  const content = (
    <div className="space-y-6">
      {/* Logging Integrations Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <CogIcon className="h-4 w-4 text-info" />
          <span className="font-semibold text-foreground">{t("loggingSettingsView.loggingIntegrations")}</span>
          <Badge variant="secondary">{loggingConfigs.length}</Badge>
        </div>

        {loggingConfigs.length > 0 ? (
          <div className="space-y-3">
            {loggingConfigs.map((config, index) => {
              const displayName = getLoggingDisplayName(config.callback_name);

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-info/10 border border-info/20"
                >
                  <div className="flex items-center gap-3">
                    <Logo
                      src={callbackInfo[displayName]?.logo}
                      label={displayName}
                      className="w-5 h-5 object-contain"
                    />
                    <div>
                      <span className="block font-medium text-info">{displayName}</span>
                      <span className="block text-xs text-info">
                        {t("loggingSettingsView.parametersConfigured", {
                          count: Object.keys(config.callback_vars).length,
                        })}
                      </span>
                    </div>
                  </div>
                  <Badge variant={getEventTypeVariant(config.callback_type)}>
                    {getEventTypeLabel(config.callback_type)}
                  </Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
            <CogIcon className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground text-sm">{t("loggingSettingsView.noLoggingIntegrations")}</span>
          </div>
        )}
      </div>

      {/* Disabled Callbacks Section */}
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <BanIcon className="h-4 w-4 text-destructive" />
          <span className="font-semibold text-foreground">{t("loggingSettingsView.disabledCallbacks")}</span>
          <Badge variant="destructive">{disabledCallbacks.length}</Badge>
        </div>

        {disabledCallbacks.length > 0 ? (
          <div className="space-y-3">
            {disabledCallbacks.map((callbackName, index) => {
              // Handle both display names and internal values
              const displayName = reverse_callback_map[callbackName] || callbackName;

              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-destructive/10 border border-destructive/20"
                >
                  <div className="flex items-center gap-3">
                    <Logo
                      src={callbackInfo[displayName]?.logo}
                      label={displayName}
                      className="w-5 h-5 object-contain"
                    />
                    <div>
                      <span className="block font-medium text-destructive">{displayName}</span>
                      <span className="block text-xs text-destructive">{t("loggingSettingsView.disabledForKey")}</span>
                    </div>
                  </div>
                  <Badge variant="destructive">{t("loggingSettingsView.disabled")}</Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
            <BanIcon className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground text-sm">{t("loggingSettingsView.noCallbacksDisabled")}</span>
          </div>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div className={`bg-card border border-border rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-6">
          <div>
            <span className="block font-semibold text-foreground">{t("loggingSettingsView.title")}</span>
            <span className="block text-xs text-muted-foreground">{t("loggingSettingsView.description")}</span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={`${className}`}>
      <span className="block font-medium text-foreground mb-3">{t("loggingSettingsView.title")}</span>
      {content}
    </div>
  );
}

export default LoggingSettingsView;
