/* eslint-disable react/no-unescaped-entities */
import React from "react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { CogIcon, BanIcon } from "@heroicons/react/outline";
import { Eye, EyeOff, Info, Plus, Trash2 } from "lucide-react";
import { callbackInfo, callback_map, mapDisplayToInternalNames } from "../callback_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";
import NumericalInput from "../shared/numerical_input";

const CALLBACK_TYPE_ITEMS = [
  { value: "success", label: "Success Only" },
  { value: "failure", label: "Failure Only" },
  { value: "success_and_failure", label: "Success & Failure" },
];

const CallbackVarInput: React.FC<{
  sensitive: boolean;
  placeholder: string;
  value: string;
  onValueChange: (value: string) => void;
}> = ({ sensitive, placeholder, value, onValueChange }) => {
  const [revealed, setRevealed] = React.useState(false);

  if (!sensitive) {
    return <Input placeholder={placeholder} value={value} onChange={(e) => onValueChange(e.target.value)} />;
  }

  return (
    <InputGroup>
      <InputGroupInput
        type={revealed ? "text" : "password"}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
      />
      <InputGroupAddon align="inline-end">
        <InputGroupButton
          size="icon-xs"
          onClick={() => setRevealed(!revealed)}
          aria-label={revealed ? "Hide password" : "Show password"}
        >
          {revealed ? <EyeOff /> : <Eye />}
        </InputGroupButton>
      </InputGroupAddon>
    </InputGroup>
  );
};

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
    // Map display names to internal callback values
    const mappedDisabledCallbacks = mapDisplayToInternalNames(newDisabledCallbacks);
    onDisabledCallbacksChange?.(mappedDisabledCallbacks);
  };

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

  const updateLoggingConfig = (index: number, field: keyof LoggingConfig, newValue: any) => {
    const updatedConfigs = [...value];
    if (field === "callback_name") {
      // Convert display name to callback value and reset callback_vars when callback changes
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

    // Find the display name for the callback
    const callbackDisplayName = Object.entries(callback_map).find(([_, value]) => value === config.callback_name)?.[0];

    if (!callbackDisplayName) return null;

    const dynamicParams = callbackInfo[callbackDisplayName]?.dynamic_params || {};

    if (Object.keys(dynamicParams).length === 0) return null;

    return (
      <div className="mt-6 pt-4 border-t border-border">
        <div className="flex items-center space-x-2 mb-4">
          <div className="w-3 h-3 bg-muted rounded-full flex items-center justify-center">
            <div className="w-1.5 h-1.5 bg-primary rounded-full"></div>
          </div>
          <span className="text-sm font-medium text-foreground">Integration Parameters</span>
        </div>
        <div className="grid grid-cols-1 gap-4">
          {Object.entries(dynamicParams).map(([paramName, paramType]) => (
            <div key={paramName} className="space-y-2">
              <label className="text-sm font-medium text-foreground capitalize flex items-center space-x-1">
                <span>{paramName.replace(/_/g, " ")}</span>
                {paramType === "password" && <Badge variant="secondary">Sensitive</Badge>}
                {paramType === "number" && <Badge variant="secondary">Number</Badge>}
              </label>
              {paramType === "number" && (
                <span className="text-xs text-muted-foreground">Value must be between 0 and 1</span>
              )}
              {paramType === "number" ? (
                <NumericalInput
                  step={0.01}
                  width={400}
                  placeholder={`os.environ/${paramName.toUpperCase()}`}
                  value={config.callback_vars[paramName] || ""}
                  onChange={(e: any) => updateCallbackVar(configIndex, paramName, e.target.value)}
                />
              ) : (
                <CallbackVarInput
                  sensitive={paramType === "password"}
                  placeholder={`os.environ/${paramName.toUpperCase()}`}
                  value={config.callback_vars[paramName] || ""}
                  onValueChange={(newValue) => updateCallbackVar(configIndex, paramName, newValue)}
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
          <BanIcon className="w-5 h-5 text-destructive" />
          <span className="text-base font-semibold text-foreground">Disabled Callbacks</span>
          <SimpleTooltip content="Select callbacks to disable for this key. Disabled callbacks will not receive any logging data.">
            <Info className="size-4 text-muted-foreground cursor-help" />
          </SimpleTooltip>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium text-foreground">Disabled Callbacks</label>
          <Select multiple value={disabledCallbacks} onValueChange={handleDisabledCallbacksChange}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select callbacks to disable">
                {(selected: string[]) => (selected.length === 0 ? "Select callbacks to disable" : selected.join(", "))}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {allCallbacks.map((callbackName) => {
                const description = callbackInfo[callbackName]?.description;
                return (
                  <SelectItem key={callbackName} value={callbackName}>
                    <SimpleTooltip content={description} side="right">
                      <div className="flex items-center space-x-2">
                        <Logo
                          src={callbackInfo[callbackName]?.logo}
                          label={callbackName}
                          className="w-4 h-4 object-contain"
                        />
                        <span>{callbackName}</span>
                      </div>
                    </SimpleTooltip>
                  </SelectItem>
                );
              })}
            </SelectContent>
          </Select>
          <div className="text-xs text-muted-foreground">
            Select callbacks that should be disabled for this key. These callbacks will not receive any logging data.
          </div>
        </div>
      </div>

      <Separator className="my-6" />

      {/* Logging Integrations Section */}
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <CogIcon className="w-5 h-5 text-foreground" />
          <span className="text-base font-semibold text-foreground">Logging Integrations</span>
          <SimpleTooltip content="Configure callback logging integrations for this team.">
            <Info className="size-4 text-muted-foreground cursor-help" />
          </SimpleTooltip>
        </div>
        <Button variant="secondary" onClick={addLoggingConfig} size="sm" type="button">
          <Plus />
          Add Integration
        </Button>
      </div>

      <div className="space-y-4">
        {value.map((config, index) => {
          const callbackDisplayName = config.callback_name
            ? Object.entries(callback_map).find(([_, value]) => value === config.callback_name)?.[0]
            : undefined;

          return (
            <Card
              key={index}
              className="block p-6 border border-border shadow-xs hover:shadow-md transition-shadow duration-200"
            >
              <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-2">
                  {callbackDisplayName && (
                    <Logo
                      src={callbackInfo[callbackDisplayName]?.logo}
                      label={callbackDisplayName}
                      className="w-5 h-5 object-contain"
                    />
                  )}
                  <span className="text-sm font-medium">{callbackDisplayName || "New Integration"} Configuration</span>
                </div>
                <Button
                  variant="ghost"
                  onClick={() => removeLoggingConfig(index)}
                  size="sm"
                  className="text-destructive hover:bg-destructive/10 hover:text-destructive/80"
                  type="button"
                >
                  <Trash2 />
                  Remove
                </Button>
              </div>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Integration Type</label>
                    <Select
                      value={callbackDisplayName ?? null}
                      onValueChange={(value: string | null) =>
                        value && updateLoggingConfig(index, "callback_name", value)
                      }
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue placeholder="Select integration" />
                      </SelectTrigger>
                      <SelectContent>
                        {supportedCallbacks.map((callbackName) => {
                          const description = callbackInfo[callbackName]?.description;
                          return (
                            <SelectItem key={callbackName} value={callbackName}>
                              <SimpleTooltip content={description} side="right">
                                <div className="flex items-center space-x-2">
                                  <Logo
                                    src={callbackInfo[callbackName]?.logo}
                                    label={callbackName}
                                    className="w-4 h-4 object-contain"
                                  />
                                  <span>{callbackName}</span>
                                </div>
                              </SimpleTooltip>
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Event Type</label>
                    <Select
                      items={CALLBACK_TYPE_ITEMS}
                      value={config.callback_type}
                      onValueChange={(value: string | null) =>
                        value && updateLoggingConfig(index, "callback_type", value)
                      }
                    >
                      <SelectTrigger aria-label="Event Type" className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {CALLBACK_TYPE_ITEMS.map((item) => (
                          <SelectItem key={item.value} value={item.value}>
                            {item.label}
                          </SelectItem>
                        ))}
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
        <div className="text-center py-12 text-muted-foreground border-2 border-dashed border-border rounded-lg bg-muted/30">
          <CogIcon className="w-12 h-12 text-muted-foreground mb-3 mx-auto" />
          <div className="text-base font-medium mb-1">No logging integrations configured</div>
          <div className="text-sm text-muted-foreground">
            Click "Add Integration" to configure logging for this team
          </div>
        </div>
      )}
    </div>
  );
};

export default LoggingSettings;
