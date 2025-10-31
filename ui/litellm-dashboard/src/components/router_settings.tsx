import React, { useState, useEffect } from "react";
import {
  Title,
  Text,
  Button,
  TextInput,
} from "@tremor/react";
import {
  getCallbacksCall,
  setCallbacksCall,
  getRouterSettingsCall,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import RoutingStrategySelector from "./router_settings/RoutingStrategySelector";
import TagFilteringToggle from "./router_settings/TagFilteringToggle";

interface RouterSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
}

interface routingStrategyArgs {
  ttl?: number;
  lowest_latency_buffer?: number;
}

const defaultLowestLatencyArgs: routingStrategyArgs = {
  ttl: 3600,
  lowest_latency_buffer: 0,
};

const RouterSettings: React.FC<RouterSettingsProps> = ({ accessToken, userRole, userID, modelData }) => {
  const [routerSettings, setRouterSettings] = useState<{ [key: string]: any }>({});
  const [selectedStrategy, setSelectedStrategy] = useState<string | null>(null);
  const [availableRoutingStrategies, setAvailableRoutingStrategies] = useState<string[]>([]);
  const [routerFieldsMetadata, setRouterFieldsMetadata] = useState<{ [key: string]: any }>({});
  const [routingStrategyDescriptions, setRoutingStrategyDescriptions] = useState<{ [key: string]: string }>({});
  const [enableTagFiltering, setEnableTagFiltering] = useState<boolean>(false);

  // Param explanations for routing strategy args (these are dynamic based on strategy)
  let paramExplanation: { [key: string]: string } = {
    ttl: "Sliding window to look back over when calculating the average latency of a deployment. Default - 1 hour (in seconds).",
    lowest_latency_buffer:
      "Shuffle between deployments within this % of the lowest latency. Default - 0 (i.e. always pick lowest latency).",
  };

  useEffect(() => {
    if (!accessToken || !userRole || !userID) {
      return;
    }
    getCallbacksCall(accessToken, userID, userRole).then((data) => {
      console.log("callbacks", data);
      let router_settings = data.router_settings;
      if ("model_group_retry_policy" in router_settings) {
        delete router_settings["model_group_retry_policy"];
      }
      setRouterSettings(router_settings);
      // Set initial selected strategy
      if (router_settings.routing_strategy) {
        setSelectedStrategy(router_settings.routing_strategy);
      }
    });
      getRouterSettingsCall(accessToken).then((data) => {
        console.log("router settings from API", data);
        if (data.fields) {
          // Build metadata map for easy lookup
          const fieldsMap: { [key: string]: any } = {};
          data.fields.forEach((field: any) => {
            fieldsMap[field.field_name] = {
              ui_field_name: field.ui_field_name,
              field_description: field.field_description,
              options: field.options,
              link: field.link,
            };
          });
          setRouterFieldsMetadata(fieldsMap);
          
          // Extract routing strategies from the routing_strategy field's options
          const routingStrategyField = data.fields.find(
            (field: any) => field.field_name === "routing_strategy"
          );
          if (routingStrategyField?.options) {
            setAvailableRoutingStrategies(routingStrategyField.options);
          }

          // Store routing strategy descriptions
          if (data.routing_strategy_descriptions) {
            setRoutingStrategyDescriptions(data.routing_strategy_descriptions);
          }

          // Set enable_tag_filtering value
          const tagFilteringField = data.fields.find(
            (field: any) => field.field_name === "enable_tag_filtering"
          );
          if (tagFilteringField?.field_value !== null && tagFilteringField?.field_value !== undefined) {
            setEnableTagFiltering(tagFilteringField.field_value);
          }
        }
      });
    }, [accessToken, userRole, userID]);

  const handleSaveChanges = (router_settings: any) => {
    if (!accessToken) {
      return;
    }

    console.log("router_settings", router_settings);

    const numberKeys = new Set(["allowed_fails", "cooldown_time", "num_retries", "timeout", "retry_after"]);
    const jsonKeys = new Set(["model_group_alias", "retry_policy"]);

    const parseInputValue = (key: string, raw: string | undefined, fallback: unknown) => {
      if (raw === undefined) return fallback;

      const v = raw.trim();

      if (v.toLowerCase() === "null") return null;

      if (numberKeys.has(key)) {
        const n = Number(v);
        return Number.isNaN(n) ? fallback : n;
      }

      if (jsonKeys.has(key)) {
        if (v === "") return null;
        try {
          return JSON.parse(v);
        } catch {
          return fallback;
        }
      }

      if (v.toLowerCase() === "true") return true;
      if (v.toLowerCase() === "false") return false;

      return v;
    };

      // Add enable_tag_filtering to router_settings before processing
      const settingsToUpdate = {
        ...router_settings,
        enable_tag_filtering: enableTagFiltering,
      };

      const updatedVariables = Object.fromEntries(
        Object.entries(settingsToUpdate)
          .map(([key, value]) => {
            if (key !== "routing_strategy_args" && key !== "routing_strategy" && key !== "enable_tag_filtering") {
              const inputEl = document.querySelector(`input[name="${key}"]`) as HTMLInputElement | null;
              const parsed = parseInputValue(key, inputEl?.value, value);
              return [key, parsed];
            } else if (key === "routing_strategy") {
              return [key, selectedStrategy];
            } else if (key === "enable_tag_filtering") {
              return [key, enableTagFiltering];
            } else if (key === "routing_strategy_args" && selectedStrategy === "latency-based-routing") {
            let setRoutingStrategyArgs: routingStrategyArgs = {};

            const lowestLatencyBufferElement = document.querySelector(
              `input[name="lowest_latency_buffer"]`,
            ) as HTMLInputElement;
            const ttlElement = document.querySelector(`input[name="ttl"]`) as HTMLInputElement;

            if (lowestLatencyBufferElement?.value) {
              setRoutingStrategyArgs["lowest_latency_buffer"] = Number(lowestLatencyBufferElement.value);
            }

            if (ttlElement?.value) {
              setRoutingStrategyArgs["ttl"] = Number(ttlElement.value);
            }

            console.log(`setRoutingStrategyArgs: ${setRoutingStrategyArgs}`);
            return ["routing_strategy_args", setRoutingStrategyArgs];
          }
          return null;
        })
        .filter((entry) => entry !== null && entry !== undefined) as Iterable<[string, unknown]>,
    );
    console.log("updatedVariables", updatedVariables);

    const payload = {
      router_settings: updatedVariables,
    };

    try {
      setCallbacksCall(accessToken, payload);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to update router settings: " + error);
    }

    NotificationsManager.success("router settings updated successfully");
  };

  if (!accessToken) {
    return null;
  }

    return (
      <div className="w-full space-y-8 py-2">
        {/* Routing Settings Section */}
        <div className="space-y-6">
          <div className="max-w-3xl">
            <h3 className="text-sm font-medium text-gray-900">Routing Settings</h3>
            <p className="text-xs text-gray-500 mt-1">Configure how requests are routed to deployments</p>
          </div>
          
          {/* Routing Strategy */}
          {routerSettings.routing_strategy && (
            <RoutingStrategySelector
              selectedStrategy={selectedStrategy || routerSettings.routing_strategy}
              availableStrategies={availableRoutingStrategies}
              routingStrategyDescriptions={routingStrategyDescriptions}
              routerFieldsMetadata={routerFieldsMetadata}
              onStrategyChange={setSelectedStrategy}
            />
          )}

          {/* Tag Filtering */}
          <TagFilteringToggle
            enabled={enableTagFiltering}
            routerFieldsMetadata={routerFieldsMetadata}
            onToggle={setEnableTagFiltering}
          />
        </div>

        {/* Divider */}
        <div className="border-t border-gray-200" />

        {/* Strategy-Specific Args - Show immediately after strategy if latency-based */}
      {selectedStrategy === "latency-based-routing" && (
        <>
          <div className="space-y-6">
            <div className="max-w-3xl">
              <h3 className="text-sm font-medium text-gray-900">Latency-Based Configuration</h3>
              <p className="text-xs text-gray-500 mt-1">Fine-tune latency-based routing behavior</p>
            </div>
            
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
              {Object.entries(
                routerSettings["routing_strategy_args"] || defaultLowestLatencyArgs
              ).map(([param, value]) => (
                <div key={param} className="space-y-2">
                  <label className="block">
                    <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
                      {param.replace(/_/g, " ")}
                    </span>
                    <p className="text-xs text-gray-500 mt-0.5 mb-2">
                      {paramExplanation[param] || ""}
                    </p>
                    <TextInput
                      name={param}
                      defaultValue={typeof value === "object" ? JSON.stringify(value, null, 2) : value?.toString()}
                      className="font-mono text-sm w-full"
                    />
                  </label>
                </div>
              ))}
            </div>
          </div>
          
          <div className="border-t border-gray-200" />
        </>
      )}

      {/* Other Settings */}
      <div className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-gray-900">Reliability & Retries</h3>
          <p className="text-xs text-gray-500 mt-1">Configure retry logic and failure handling</p>
        </div>
        
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
          {Object.entries(routerSettings)
            .filter(
              ([param, value]) =>
                param != "fallbacks" &&
                param != "context_window_fallbacks" &&
                param != "routing_strategy_args" &&
                param != "routing_strategy" &&
                param != "enable_tag_filtering",
            )
            .map(([param, value]) => (
              <div key={param} className="space-y-2">
                <label className="block">
                  <span className="text-xs font-medium text-gray-700 uppercase tracking-wide">
                    {routerFieldsMetadata[param]?.ui_field_name || param}
                  </span>
                  <p className="text-xs text-gray-500 mt-0.5 mb-2">
                    {routerFieldsMetadata[param]?.field_description || ""}
                  </p>
                  <TextInput
                    name={param}
                    defaultValue={
                      typeof value === "object" ? JSON.stringify(value, null, 2) : value?.toString() || ""
                    }
                    placeholder="—"
                    className="font-mono text-sm w-full"
                  />
                </label>
              </div>
            ))}
        </div>
      </div>

      {/* Actions - Sticky at bottom */}
      <div className="border-t border-gray-200 pt-6 flex justify-end gap-3">
        <Button 
          variant="secondary"
          size="sm"
          onClick={() => window.location.reload()}
          className="text-sm"
        >
          Reset
        </Button>
        <Button 
          size="sm"
          onClick={() => handleSaveChanges(routerSettings)}
          className="text-sm font-medium"
        >
          Save Changes
        </Button>
      </div>
    </div>
  );
};

export default RouterSettings;

