import React from "react";
import { TextInput } from "@tremor/react";

interface ReliabilityRetriesSectionProps {
  routerSettings: { [key: string]: any };
  routerFieldsMetadata: { [key: string]: any };
}

const ReliabilityRetriesSection: React.FC<ReliabilityRetriesSectionProps> = ({
  routerSettings,
  routerFieldsMetadata,
}) => {
  return (
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
                    value === null || value === undefined || value === "null"
                      ? ""
                      : typeof value === "object"
                        ? JSON.stringify(value, null, 2)
                        : value?.toString() || ""
                  }
                  placeholder="—"
                  className="font-mono text-sm w-full"
                />
              </label>
            </div>
          ))}
      </div>
    </div>
  );
};

export default ReliabilityRetriesSection;

