import React from "react";
import LatencyBasedConfiguration from "./LatencyBasedConfiguration";
import ReliabilityRetriesSection from "./ReliabilityRetriesSection";
import RoutingStrategySelector from "./RoutingStrategySelector";
import TagFilteringToggle from "./TagFilteringToggle";

export interface RouterSettingsFormValue {
  routerSettings: { [key: string]: any };
  selectedStrategy: string | null;
  enableTagFiltering: boolean;
}

interface RouterSettingsFormProps {
  value: RouterSettingsFormValue;
  onChange: (value: RouterSettingsFormValue) => void;
  routerFieldsMetadata: { [key: string]: any };
  availableRoutingStrategies: string[];
  routingStrategyDescriptions: { [key: string]: string };
}

const RouterSettingsForm: React.FC<RouterSettingsFormProps> = ({
  value,
  onChange,
  routerFieldsMetadata,
  availableRoutingStrategies,
  routingStrategyDescriptions,
}) => {
  const handleStrategyChange = (strategy: string) => {
    onChange({
      ...value,
      selectedStrategy: strategy,
    });
  };

  const handleTagFilteringToggle = (enabled: boolean) => {
    onChange({
      ...value,
      enableTagFiltering: enabled,
    });
  };

  return (
    <div className="w-full space-y-8 py-2">
      {/* Routing Settings Section */}
      <div className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-gray-900">Routing Settings</h3>
          <p className="text-xs text-gray-500 mt-1">Configure how requests are routed to deployments</p>
        </div>

        {/* Routing Strategy */}
        {availableRoutingStrategies.length > 0 && (
          <RoutingStrategySelector
            selectedStrategy={value.selectedStrategy || value.routerSettings.routing_strategy || null}
            availableStrategies={availableRoutingStrategies}
            routingStrategyDescriptions={routingStrategyDescriptions}
            routerFieldsMetadata={routerFieldsMetadata}
            onStrategyChange={handleStrategyChange}
          />
        )}

        {/* Tag Filtering */}
        <TagFilteringToggle
          enabled={value.enableTagFiltering}
          routerFieldsMetadata={routerFieldsMetadata}
          onToggle={handleTagFilteringToggle}
        />
      </div>

      {/* Divider */}
      <div className="border-t border-gray-200" />

      {/* Strategy-Specific Args - Show immediately after strategy if latency-based */}
      {value.selectedStrategy === "latency-based-routing" && (
        <LatencyBasedConfiguration routingStrategyArgs={value.routerSettings["routing_strategy_args"]} />
      )}

      {/* Other Settings */}
      <ReliabilityRetriesSection routerSettings={value.routerSettings} routerFieldsMetadata={routerFieldsMetadata} />
    </div>
  );
};

export default RouterSettingsForm;
