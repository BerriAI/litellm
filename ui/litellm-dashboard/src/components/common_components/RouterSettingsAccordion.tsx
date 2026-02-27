import React, { useEffect, useState, useImperativeHandle, forwardRef, useRef } from "react";
import { TabPanel, TabPanels, TabGroup, TabList, Tab } from "@tremor/react";
import { getRouterSettingsCall } from "../networking";
import RouterSettingsForm, { RouterSettingsFormValue } from "../router_settings/RouterSettingsForm";
import { Fallbacks } from "../Settings/RouterSettings/Fallbacks/AddFallbacks";
import { FallbackSelectionForm } from "../Settings/RouterSettings/Fallbacks/FallbackSelectionForm";
import { FallbackGroup } from "../Settings/RouterSettings/Fallbacks/FallbackGroupConfig";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

export interface RouterSettingsAccordionValue {
  router_settings: {
    routing_strategy?: string | null;
    allowed_fails?: number | null;
    cooldown_time?: number | null;
    num_retries?: number | null;
    timeout?: number | null;
    retry_after?: number | null;
    fallbacks?: Fallbacks | null;
    context_window_fallbacks?: any | null;
    retry_policy?: any | null;
    model_group_alias?: { [key: string]: any } | null;
    enable_tag_filtering?: boolean;
    routing_strategy_args?: { [key: string]: any } | null;
  };
}

interface RouterSettingsAccordionProps {
  accessToken: string;
  value?: RouterSettingsAccordionValue;
  onChange?: (value: RouterSettingsAccordionValue) => void;
  modelData?: any;
}

export interface RouterSettingsAccordionRef {
  getValue: () => RouterSettingsAccordionValue;
}

const RouterSettingsAccordion = forwardRef<RouterSettingsAccordionRef, RouterSettingsAccordionProps>(
  ({ accessToken, value, onChange, modelData }, ref) => {
    const [formValue, setFormValue] = useState<RouterSettingsFormValue>({
      routerSettings: {},
      selectedStrategy: null,
      enableTagFiltering: false,
    });
    const [fallbacks, setFallbacks] = useState<Fallbacks>([]);
    const [fallbackGroups, setFallbackGroups] = useState<FallbackGroup[]>([]);
    const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
    const [availableRoutingStrategies, setAvailableRoutingStrategies] = useState<string[]>([]);
    const [routerFieldsMetadata, setRouterFieldsMetadata] = useState<{ [key: string]: any }>({});
    const [routingStrategyDescriptions, setRoutingStrategyDescriptions] = useState<{ [key: string]: string }>({});
    const isInternalUpdateRef = useRef(false);
    const lastInitializedValueRef = useRef<string | null>(null);

    // Convert fallbacks format to groups format
    const fallbacksToGroups = (fallbacks: Fallbacks): FallbackGroup[] => {
      if (!fallbacks || fallbacks.length === 0) {
        return [
          {
            id: "1",
            primaryModel: null,
            fallbackModels: [],
          },
        ];
      }
      return fallbacks.map((entry, index) => {
        const [primaryModel, fallbackModels] = Object.entries(entry)[0];
        return {
          id: (index + 1).toString(),
          primaryModel: primaryModel || null,
          fallbackModels: fallbackModels || [],
        };
      });
    };

    // Convert groups format to fallbacks format
    const groupsToFallbacks = (groups: FallbackGroup[]): Fallbacks => {
      return groups
        .filter((g) => g.primaryModel && g.fallbackModels.length > 0)
        .map((g) => ({
          [g.primaryModel!]: g.fallbackModels,
        }));
    };

    // Initialize from value prop if provided (only when value actually changes externally)
    useEffect(() => {
      // Create a stable key from the value to detect actual external changes
      const valueKey = value?.router_settings
        ? JSON.stringify({
          routing_strategy: value.router_settings.routing_strategy,
          fallbacks: value.router_settings.fallbacks,
          enable_tag_filtering: value.router_settings.enable_tag_filtering,
        })
        : null;

      // Skip if this is an internal update (from our own onChange) and the value hasn't actually changed
      if (isInternalUpdateRef.current && valueKey === lastInitializedValueRef.current) {
        isInternalUpdateRef.current = false;
        return;
      }

      // Reset the flag if value actually changed externally
      if (isInternalUpdateRef.current && valueKey !== lastInitializedValueRef.current) {
        isInternalUpdateRef.current = false;
      }

      // Only update if the value actually changed from an external source
      if (valueKey === lastInitializedValueRef.current) {
        return;
      }

      lastInitializedValueRef.current = valueKey;

      if (value?.router_settings) {
        const rs = value.router_settings;
        const { fallbacks: _, ...routerSettingsWithoutFallbacks } = rs;
        setFormValue({
          routerSettings: routerSettingsWithoutFallbacks as { [key: string]: any },
          selectedStrategy: rs.routing_strategy || null,
          enableTagFiltering: rs.enable_tag_filtering ?? false,
        });
        const initialFallbacks = rs.fallbacks || [];
        setFallbacks(initialFallbacks);
        setFallbackGroups(fallbacksToGroups(initialFallbacks));
      } else {
        // Initialize with empty defaults if no value provided
        setFormValue({
          routerSettings: {},
          selectedStrategy: null,
          enableTagFiltering: false,
        });
        setFallbacks([]);
        setFallbackGroups([
          {
            id: "1",
            primaryModel: null,
            fallbackModels: [],
          },
        ]);
      }
    }, [value]);

    // Fetch router settings metadata
    useEffect(() => {
      if (!accessToken) {
        return;
      }
      getRouterSettingsCall(accessToken).then((data) => {
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
          const routingStrategyField = data.fields.find((field: any) => field.field_name === "routing_strategy");
          if (routingStrategyField?.options) {
            setAvailableRoutingStrategies(routingStrategyField.options);
          }

          // Store routing strategy descriptions
          if (data.routing_strategy_descriptions) {
            setRoutingStrategyDescriptions(data.routing_strategy_descriptions);
          }
        }
      });
    }, [accessToken]);

    // Fetch available models for fallbacks
    useEffect(() => {
      if (!accessToken) {
        return;
      }
      const loadModels = async () => {
        try {
          const uniqueModels = await fetchAvailableModels(accessToken);
          setModelInfo(uniqueModels);
        } catch (error) {
          console.error("Error fetching model info for fallbacks:", error);
        }
      };
      loadModels();
    }, [accessToken]);

    // Helper function to build router_settings from current state
    const buildRouterSettings = (): RouterSettingsAccordionValue["router_settings"] => {
      // Parse input values from DOM (similar to RouterSettings component)
      const numberKeys = new Set(["allowed_fails", "cooldown_time", "num_retries", "timeout", "retry_after"]);
      const jsonKeys = new Set(["model_group_alias", "retry_policy"]);

      const parseInputValue = (key: string, raw: string | undefined, fallback: unknown) => {
        if (raw === undefined || raw === null) return fallback;

        const v = String(raw).trim();

        if (v === "" || v.toLowerCase() === "null") return null;

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

      // Build router_settings object
      const routerSettings = formValue.routerSettings;
      const settingsToUpdate: { [key: string]: any } = {
        ...routerSettings,
        enable_tag_filtering: formValue.enableTagFiltering,
        routing_strategy: formValue.selectedStrategy,
        fallbacks: fallbacks.length > 0 ? fallbacks : null,
      };

      // Parse values from DOM inputs for reliability/retry fields
      const updatedVariables = Object.fromEntries(
        Object.entries(settingsToUpdate)
          .map(([key, value]) => {
            if (
              key !== "routing_strategy_args" &&
              key !== "routing_strategy" &&
              key !== "enable_tag_filtering" &&
              key !== "fallbacks"
            ) {
              const inputEl = document.querySelector(`input[name="${key}"]`) as HTMLInputElement | null;
              if (inputEl && inputEl.value !== undefined && inputEl.value !== "") {
                const parsed = parseInputValue(key, inputEl.value, value);
                return [key, parsed];
              }
              return [key, value];
            } else if (key === "routing_strategy") {
              return [key, formValue.selectedStrategy];
            } else if (key === "enable_tag_filtering") {
              return [key, formValue.enableTagFiltering];
            } else if (key === "fallbacks") {
              return [key, fallbacks.length > 0 ? fallbacks : null];
            } else if (key === "routing_strategy_args" && formValue.selectedStrategy === "latency-based-routing") {
              const lowestLatencyBufferElement = document.querySelector(
                `input[name="lowest_latency_buffer"]`,
              ) as HTMLInputElement;
              const ttlElement = document.querySelector(`input[name="ttl"]`) as HTMLInputElement;

              const routingStrategyArgs: { [key: string]: any } = {};
              if (lowestLatencyBufferElement?.value) {
                routingStrategyArgs["lowest_latency_buffer"] = Number(lowestLatencyBufferElement.value);
              }
              if (ttlElement?.value) {
                routingStrategyArgs["ttl"] = Number(ttlElement.value);
              }
              return ["routing_strategy_args", Object.keys(routingStrategyArgs).length > 0 ? routingStrategyArgs : null];
            }
            return [key, value];
          })
          .filter((entry) => entry !== null && entry !== undefined) as Iterable<[string, unknown]>,
      );

      // Ensure all required fields are present with null defaults if not set
      // Convert empty objects/undefined to null for proper type safety
      const normalizeValue = (val: any, isNumber = false): any => {
        if (val === undefined || val === null) return null;
        if (typeof val === "object" && !Array.isArray(val) && Object.keys(val).length === 0) return null;
        if (isNumber && (typeof val !== "number" || Number.isNaN(val))) return null;
        return val;
      };

      return {
        routing_strategy: normalizeValue(updatedVariables.routing_strategy) as string | null | undefined,
        allowed_fails: normalizeValue(updatedVariables.allowed_fails, true) as number | null | undefined,
        cooldown_time: normalizeValue(updatedVariables.cooldown_time, true) as number | null | undefined,
        num_retries: normalizeValue(updatedVariables.num_retries, true) as number | null | undefined,
        timeout: normalizeValue(updatedVariables.timeout, true) as number | null | undefined,
        retry_after: normalizeValue(updatedVariables.retry_after, true) as number | null | undefined,
        fallbacks: fallbacks.length > 0 ? fallbacks : null,
        context_window_fallbacks: normalizeValue(updatedVariables.context_window_fallbacks),
        retry_policy: normalizeValue(updatedVariables.retry_policy),
        model_group_alias: normalizeValue(updatedVariables.model_group_alias),
        enable_tag_filtering: formValue.enableTagFiltering,
        routing_strategy_args: normalizeValue(updatedVariables.routing_strategy_args),
      };
    };

    // Update parent when form values change (with debounce to avoid infinite loops)
    useEffect(() => {
      if (!onChange) {
        return;
      }

      const timeoutId = setTimeout(() => {
        isInternalUpdateRef.current = true;
        const finalRouterSettings = buildRouterSettings();
        onChange({
          router_settings: finalRouterSettings,
        });
      }, 100);

      return () => clearTimeout(timeoutId);
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [formValue, fallbacks]);

    const handleFallbackGroupsChange = (newGroups: FallbackGroup[]): void => {
      setFallbackGroups(newGroups);
      const newFallbacks = groupsToFallbacks(newGroups);
      setFallbacks(newFallbacks);
    };

    const availableModels = Array.from(new Set(modelInfo.map((option) => option.model_group))).sort();

    // Expose getValue method via ref
    useImperativeHandle(ref, () => ({
      getValue: () => {
        return {
          router_settings: buildRouterSettings(),
        };
      },
    }));

    if (!accessToken) {
      return null;
    }

    return (
      <div className="w-full">
        <TabGroup className="w-full">
          <TabList variant="line" defaultValue="1" className="px-8 pt-4">
            <Tab value="1">Loadbalancing</Tab>
            <Tab value="2">Fallbacks</Tab>
          </TabList>
          <TabPanels className="px-8 py-6">
            <TabPanel>
              <RouterSettingsForm
                value={formValue}
                onChange={setFormValue}
                routerFieldsMetadata={routerFieldsMetadata}
                availableRoutingStrategies={availableRoutingStrategies}
                routingStrategyDescriptions={routingStrategyDescriptions}
              />
            </TabPanel>
            <TabPanel>
              <FallbackSelectionForm
                groups={fallbackGroups}
                onGroupsChange={handleFallbackGroupsChange}
                availableModels={availableModels}
                maxGroups={5}
              />
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </div>
    );
  });

RouterSettingsAccordion.displayName = "RouterSettingsAccordion";

export default RouterSettingsAccordion;
