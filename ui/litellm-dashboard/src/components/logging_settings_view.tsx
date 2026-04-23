import React from "react";
import { Badge } from "@/components/ui/badge";
import { Ban, Settings as Cog } from "lucide-react";
import {
  callbackInfo,
  callback_map,
  reverse_callback_map,
} from "./callback_info_helpers";
import { cn } from "@/lib/utils";

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
    const callbackDisplayName = Object.entries(callback_map).find(
      ([, value]) => value === callbackName,
    )?.[0];
    return callbackDisplayName || callbackName;
  };

  const getEventTypeBadgeClass = (eventType: string): string => {
    switch (eventType) {
      case "success":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300";
      case "failure":
        return "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300";
      case "success_and_failure":
        return "bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300";
      default:
        return "";
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
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Cog className="h-4 w-4 text-blue-600" />
          <span className="font-semibold text-foreground">
            Logging Integrations
          </span>
          <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300">
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
                  className="flex items-center justify-between p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800"
                >
                  <div className="flex items-center gap-3">
                    {logoUrl ? (
                      <img
                        src={logoUrl}
                        alt={displayName}
                        className="w-5 h-5 object-contain"
                      />
                    ) : (
                      <Cog className="h-5 w-5 text-muted-foreground" />
                    )}
                    <div>
                      <span className="block font-medium text-blue-800 dark:text-blue-300">
                        {displayName}
                      </span>
                      <span className="block text-xs text-blue-600 dark:text-blue-400">
                        {Object.keys(config.callback_vars).length} parameters
                        configured
                      </span>
                    </div>
                  </div>
                  <Badge className={getEventTypeBadgeClass(config.callback_type)}>
                    {getEventTypeLabel(config.callback_type)}
                  </Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
            <Cog className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground text-sm">
              No logging integrations configured
            </span>
          </div>
        )}
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <Ban className="h-4 w-4 text-red-600" />
          <span className="font-semibold text-foreground">
            Disabled Callbacks
          </span>
          <Badge variant="destructive">{disabledCallbacks.length}</Badge>
        </div>

        {disabledCallbacks.length > 0 ? (
          <div className="space-y-3">
            {disabledCallbacks.map((callbackName, index) => {
              const displayName =
                reverse_callback_map[callbackName] || callbackName;
              const logoUrl = callbackInfo[displayName]?.logo;
              return (
                <div
                  key={index}
                  className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800"
                >
                  <div className="flex items-center gap-3">
                    {logoUrl ? (
                      <img
                        src={logoUrl}
                        alt={displayName}
                        className="w-5 h-5 object-contain"
                      />
                    ) : (
                      <Ban className="h-5 w-5 text-muted-foreground" />
                    )}
                    <div>
                      <span className="block font-medium text-red-800 dark:text-red-300">
                        {displayName}
                      </span>
                      <span className="block text-xs text-red-600 dark:text-red-400">
                        Disabled for this key
                      </span>
                    </div>
                  </div>
                  <Badge variant="destructive">Disabled</Badge>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
            <Ban className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground text-sm">
              No callbacks disabled
            </span>
          </div>
        )}
      </div>
    </div>
  );

  if (variant === "card") {
    return (
      <div
        className={cn(
          "bg-background border border-border rounded-lg p-6",
          className,
        )}
      >
        <div className="flex items-center gap-2 mb-6">
          <div>
            <span className="block font-semibold text-foreground">
              Logging Settings
            </span>
            <span className="block text-xs text-muted-foreground">
              Active logging integrations and disabled callbacks for this key
            </span>
          </div>
        </div>
        {content}
      </div>
    );
  }

  return (
    <div className={className}>
      <span className="block font-medium text-foreground mb-3">
        Logging Settings
      </span>
      {content}
    </div>
  );
}

export default LoggingSettingsView;
