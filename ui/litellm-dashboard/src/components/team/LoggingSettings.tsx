/* eslint-disable @next/next/no-img-element */
/* eslint-disable react/no-unescaped-entities */
import React from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Ban, Info, Plus, Settings, Trash2, X } from "lucide-react";
import { callbackInfo, callback_map, mapDisplayToInternalNames } from "../callback_info_helpers";

interface LoggingConfig {
  callback_name: string;
  callback_type: string;
  callback_vars: Record<string, string>;
}

interface LoggingSettingsProps {
  value?: LoggingConfig[];
  onChange?: (value: LoggingConfig[]) => void;
  disabledCallbacks?: string[];
  onDisabledCallbacksChange?: (disabledCallbacks: string[]) => void;
}

function CallbackLogo({ name, size = "sm" }: { name: string; size?: "sm" | "md" }) {
  const logo = callbackInfo[name]?.logo;
  const dims = size === "md" ? "w-5 h-5" : "w-4 h-4";
  if (!logo) {
    return (
      <div className={`${dims} rounded-full bg-muted flex items-center justify-center text-xs`}>{name.charAt(0)}</div>
    );
  }
  return (
    <img
      src={logo}
      alt={name}
      className={`${dims} object-contain`}
      onError={(e) => {
        const target = e.target as HTMLImageElement;
        const parent = target.parentElement;
        if (parent) {
          const fallbackDiv = document.createElement("div");
          fallbackDiv.className = `${dims} rounded-full bg-muted flex items-center justify-center text-xs`;
          fallbackDiv.textContent = name.charAt(0);
          parent.replaceChild(fallbackDiv, target);
        }
      }}
    />
  );
}

const LoggingSettings: React.FC<LoggingSettingsProps> = ({
  value = [],
  onChange,
  disabledCallbacks = [],
  onDisabledCallbacksChange,
}) => {
  // Get callbacks that support team and key logging
  const supportedCallbacks = Object.entries(callbackInfo)
    .filter(([_, info]) => info.supports_key_team_logging)
    .map(([name, _]) => name);

  // Get all available callbacks for disabled selection
  const allCallbacks = Object.keys(callbackInfo);

  const handleChange = (newValue: LoggingConfig[]) => {
    onChange?.(newValue);
  };

  const handleDisabledCallbacksChange = (newDisabledCallbacks: string[]) => {
    const mappedDisabledCallbacks = mapDisplayToInternalNames(newDisabledCallbacks);
    onDisabledCallbacksChange?.(mappedDisabledCallbacks);
  };

  const addDisabledCallback = (name: string) => {
    if (!name || disabledCallbacks.includes(name)) return;
    handleDisabledCallbacksChange([...disabledCallbacks, name]);
  };

  const removeDisabledCallback = (name: string) => {
    handleDisabledCallbacksChange(disabledCallbacks.filter((n) => n !== name));
  };

  const remainingDisabledOptions = allCallbacks.filter((c) => !disabledCallbacks.includes(c));

  const addLoggingConfig = () => {
    const newConfig: LoggingConfig = {
      callback_name: "",
      callback_type: "success",
      callback_vars: {},
    };
    handleChange([...value, newConfig]);
  };

  const removeLoggingConfig = (index: number) => {
    const newValue = value.filter((_, i) => i !== index);
    handleChange(newValue);
  };

  const updateLoggingConfig = (
    index: number,
    field: keyof LoggingConfig,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    newValue: any,
  ) => {
    const updatedConfigs = [...value];
    if (field === "callback_name") {
      const callbackValue = callback_map[newValue] || newValue;
      updatedConfigs[index] = {
        ...updatedConfigs[index],
        [field]: callbackValue,
        callback_vars: {},
      };
    } else {
      updatedConfigs[index] = {
        ...updatedConfigs[index],
        [field]: newValue,
      };
    }
    handleChange(updatedConfigs);
  };

  const updateCallbackVar = (configIndex: number, varName: string, varValue: string) => {
    const updatedConfigs = [...value];
    updatedConfigs[configIndex] = {
      ...updatedConfigs[configIndex],
      callback_vars: {
        ...updatedConfigs[configIndex].callback_vars,
        [varName]: varValue,
      },
    };
    handleChange(updatedConfigs);
  };

  const renderDynamicParams = (config: LoggingConfig, configIndex: number) => {
    if (!config.callback_name) return null;

    const callbackDisplayName = Object.entries(callback_map).find(([_, v]) => v === config.callback_name)?.[0];

    if (!callbackDisplayName) return null;

    const dynamicParams = callbackInfo[callbackDisplayName]?.dynamic_params || {};

    if (Object.keys(dynamicParams).length === 0) return null;

    return (
      <div className="mt-6 pt-4 border-t border-border">
        <div className="flex items-center space-x-2 mb-4">
          <div className="w-3 h-3 bg-blue-100 dark:bg-blue-950/30 rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full"></div>
          </div>
          <span className="text-sm font-medium text-foreground">Integration Parameters</span>
        </div>
        <div className="grid grid-cols-1 gap-4">
          {Object.entries(dynamicParams).map(([paramName, paramType]) => (
            <div key={paramName} className="space-y-2">
              <label className="text-sm font-medium text-foreground capitalize flex items-center space-x-1">
                <span>{paramName.replace(/_/g, " ")}</span>
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Info className="h-3 w-3 text-muted-foreground cursor-help" />
                    </TooltipTrigger>
                    <TooltipContent>
                      Environment variable reference recommended: os.environ/{paramName.toUpperCase()}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                {paramType === "password" && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                    Sensitive
                  </span>
                )}
                {paramType === "number" && (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-amber-100 text-amber-800 dark:bg-amber-950/30 dark:text-amber-300">
                    Number
                  </span>
                )}
              </label>
              {paramType === "number" && (
                <span className="text-xs text-muted-foreground">Value must be between 0 and 1</span>
              )}
              {paramType === "number" ? (
                <Input
                  type="number"
                  step={0.01}
                  placeholder={`os.environ/${paramName.toUpperCase()}`}
                  value={config.callback_vars[paramName] || ""}
                  onChange={(e) => updateCallbackVar(configIndex, paramName, e.target.value)}
                  onWheel={(event) => (event.currentTarget as HTMLInputElement).blur()}
                />
              ) : (
                <Input
                  type={paramType === "password" ? "password" : "text"}
                  placeholder={`os.environ/${paramName.toUpperCase()}`}
                  value={config.callback_vars[paramName] || ""}
                  onChange={(e) => updateCallbackVar(configIndex, paramName, e.target.value)}
                />
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Disabled Callbacks Section */}
      <div className="space-y-4">
        <div className="flex items-center space-x-2">
          <Ban className="w-5 h-5 text-destructive" />
          <span className="text-base font-semibold text-foreground">Disabled Callbacks</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Select callbacks to disable for this key. Disabled callbacks will not receive any logging data.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Disabled Callbacks</label>
          <Select value="" onValueChange={(v) => addDisabledCallback(v)}>
            <SelectTrigger>
              <SelectValue placeholder="Select callbacks to disable" />
            </SelectTrigger>
            <SelectContent>
              {remainingDisabledOptions.length === 0 ? (
                <div className="py-2 px-3 text-sm text-muted-foreground">No more callbacks available</div>
              ) : (
                remainingDisabledOptions.map((callbackName) => (
                  <SelectItem key={callbackName} value={callbackName}>
                    <div className="flex items-center space-x-2">
                      <CallbackLogo name={callbackName} />
                      <span>{callbackName}</span>
                    </div>
                  </SelectItem>
                ))
              )}
            </SelectContent>
          </Select>
          {disabledCallbacks.length > 0 && (
            <div className="flex flex-wrap gap-1 pt-1">
              {disabledCallbacks.map((name) => (
                <Badge key={name} variant="secondary" className="flex items-center gap-1">
                  {name}
                  <button
                    type="button"
                    onClick={() => removeDisabledCallback(name)}
                    className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                    aria-label={`Remove ${name}`}
                  >
                    <X size={12} />
                  </button>
                </Badge>
              ))}
            </div>
          )}
          <div className="text-xs text-muted-foreground">
            Select callbacks that should be disabled for this key. These callbacks will not receive any logging data.
          </div>
        </div>
      </div>

      <hr className="my-4 border-border" />

      {/* Logging Integrations Section */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <Settings className="w-5 h-5 text-blue-500" />
          <span className="text-base font-semibold text-foreground">Logging Integrations</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent>Configure callback logging integrations for this team.</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <Button
          variant="secondary"
          onClick={addLoggingConfig}
          size="sm"
          className="hover:border-blue-400 hover:text-blue-500"
          type="button"
        >
          <Plus className="h-4 w-4" />
          Add Integration
        </Button>
      </div>

      <div className="space-y-4">
        {value.map((config, index) => {
          const callbackDisplayName = config.callback_name
            ? Object.entries(callback_map).find(([_, v]) => v === config.callback_name)?.[0]
            : undefined;

          return (
            <Card
              key={index}
              className="border border-border shadow-sm hover:shadow-md transition-shadow duration-200 p-4 border-t-2 border-t-blue-500"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-2">
                  {callbackDisplayName && <CallbackLogo name={callbackDisplayName} size="md" />}
                  <span className="text-sm font-medium">{callbackDisplayName || "New Integration"} Configuration</span>
                </div>
                <Button
                  variant="ghost"
                  onClick={() => removeLoggingConfig(index)}
                  size="sm"
                  className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  type="button"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  Remove
                </Button>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Integration Type</label>
                    <Select
                      value={callbackDisplayName || ""}
                      onValueChange={(v) => updateLoggingConfig(index, "callback_name", v)}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select integration" />
                      </SelectTrigger>
                      <SelectContent>
                        {supportedCallbacks.map((callbackName) => (
                          <SelectItem key={callbackName} value={callbackName}>
                            <div className="flex items-center space-x-2">
                              <CallbackLogo name={callbackName} />
                              <span>{callbackName}</span>
                            </div>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Event Type</label>
                    <Select
                      value={config.callback_type}
                      onValueChange={(v) => updateLoggingConfig(index, "callback_type", v)}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="success">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
                            <span>Success Only</span>
                          </div>
                        </SelectItem>
                        <SelectItem value="failure">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                            <span>Failure Only</span>
                          </div>
                        </SelectItem>
                        <SelectItem value="success_and_failure">
                          <div className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                            <span>Success & Failure</span>
                          </div>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                {renderDynamicParams(config, index)}
              </div>
            </Card>
          );
        })}
      </div>

      {value.length === 0 && (
        <div className="text-center py-12 text-muted-foreground border-2 border-dashed border-border rounded-lg bg-muted/50">
          <Settings className="w-12 h-12 text-muted-foreground/60 mb-3 mx-auto" />
          <div className="text-base font-medium mb-1">No logging integrations configured</div>
          <div className="text-sm text-muted-foreground/80">
            Click "Add Integration" to configure logging for this team
          </div>
        </div>
      )}
    </div>
  );
};

export default LoggingSettings;
