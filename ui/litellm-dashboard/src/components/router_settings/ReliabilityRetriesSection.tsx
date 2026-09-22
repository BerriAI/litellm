import React from "react";
import { ConfigOwnedField, type FieldSourceMap, isConfigOwned } from "@/components/shared/ConfigOwnedField";
import { Input } from "@/components/ui/input";

interface ReliabilityRetriesSectionProps {
  routerSettings: { [key: string]: any };
  routerFieldsMetadata: { [key: string]: any };
  routerSources?: FieldSourceMap;
}

const ReliabilityRetriesSection: React.FC<ReliabilityRetriesSectionProps> = ({
  routerSettings,
  routerFieldsMetadata,
  routerSources = {},
}) => {
  return (
    <div className="space-y-6">
      <div className="max-w-3xl">
        <h3 className="text-sm font-medium text-foreground">Reliability & Retries</h3>
        <p className="text-xs text-muted-foreground mt-1">Configure retry logic and failure handling</p>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
        {Object.entries(routerSettings)
          .filter(
            ([param]) =>
              param != "fallbacks" &&
              param != "context_window_fallbacks" &&
              param != "routing_strategy_args" &&
              param != "routing_strategy" &&
              param != "enable_tag_filtering" &&
              param != "retry_policy" &&
              param != "model_group_retry_policy" &&
              param != "routing_groups",
          )
          .map(([param, value]) => (
            <div key={param} className="space-y-2">
              <label className="block">
                <span className="text-xs font-medium text-foreground uppercase tracking-wide">
                  {routerFieldsMetadata[param]?.ui_field_name || param}
                </span>
                <p className="text-xs text-muted-foreground mt-0.5 mb-2">
                  {routerFieldsMetadata[param]?.field_description || ""}
                </p>
                <ConfigOwnedField frozen={isConfigOwned(routerSources, param)}>
                  <Input
                    name={param}
                    disabled={isConfigOwned(routerSources, param)}
                    defaultValue={
                      value === null || value === undefined || value === "null"
                        ? ""
                        : typeof value === "object"
                          ? JSON.stringify(value, null, 2)
                          : value?.toString() || ""
                    }
                    placeholder="—"
                    className="font-mono text-sm w-full"
                  />
                </ConfigOwnedField>
              </label>
            </div>
          ))}
      </div>
    </div>
  );
};

export default ReliabilityRetriesSection;
