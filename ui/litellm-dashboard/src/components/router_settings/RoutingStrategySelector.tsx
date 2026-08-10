import React from "react";
import { Select } from "antd";

interface RoutingStrategySelectorProps {
  selectedStrategy: string | null;
  availableStrategies: string[];
  routingStrategyDescriptions: { [key: string]: string };
  routerFieldsMetadata: { [key: string]: any };
  onStrategyChange: (strategy: string) => void;
  label?: string;
  description?: string;
}

const RoutingStrategySelector: React.FC<RoutingStrategySelectorProps> = ({
  selectedStrategy,
  availableStrategies,
  routingStrategyDescriptions,
  routerFieldsMetadata,
  onStrategyChange,
  label,
  description,
}) => {
  return (
    <div className="space-y-2 max-w-3xl">
      <div>
        <label className="text-xs font-medium text-gray-700 uppercase tracking-wide">
          {label || routerFieldsMetadata["routing_strategy"]?.ui_field_name || "Routing Strategy"}
        </label>
        <p className="text-xs text-gray-500 mt-0.5 mb-2">
          {description || routerFieldsMetadata["routing_strategy"]?.field_description || ""}
        </p>
      </div>
      <div className="routing-strategy-select max-w-3xl">
        <Select value={selectedStrategy} onChange={onStrategyChange} style={{ width: "100%" }} size="large">
          {availableStrategies.map((strategy) => (
            <Select.Option key={strategy} value={strategy} label={strategy}>
              <div className="flex flex-col gap-0.5 py-1">
                <span className="font-mono text-sm font-medium">{strategy}</span>
                {routingStrategyDescriptions[strategy] && (
                  <span className="text-xs text-gray-500 font-normal">{routingStrategyDescriptions[strategy]}</span>
                )}
              </div>
            </Select.Option>
          ))}
        </Select>
      </div>
    </div>
  );
};

export default RoutingStrategySelector;
