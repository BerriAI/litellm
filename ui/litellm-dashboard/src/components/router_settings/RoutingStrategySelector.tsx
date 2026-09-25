import React from "react";
import { ConfigOwnedField } from "@/components/shared/ConfigOwnedField";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface RoutingStrategySelectorProps {
  selectedStrategy: string | null;
  availableStrategies: string[];
  routingStrategyDescriptions: { [key: string]: string };
  routerFieldsMetadata: { [key: string]: any };
  disabled?: boolean;
  onStrategyChange: (strategy: string) => void;
}

const RoutingStrategySelector: React.FC<RoutingStrategySelectorProps> = ({
  selectedStrategy,
  availableStrategies,
  routingStrategyDescriptions,
  routerFieldsMetadata,
  disabled = false,
  onStrategyChange,
}) => {
  return (
    <div className="space-y-2 max-w-3xl">
      <div>
        <label className="text-xs font-medium text-foreground uppercase tracking-wide">
          {routerFieldsMetadata["routing_strategy"]?.ui_field_name || "Routing Strategy"}
        </label>
        <p className="text-xs text-muted-foreground mt-0.5 mb-2">
          {routerFieldsMetadata["routing_strategy"]?.field_description || ""}
        </p>
      </div>
      <ConfigOwnedField frozen={disabled} className="block max-w-3xl">
        <Select
          value={selectedStrategy}
          disabled={disabled}
          onValueChange={(strategy: string | null) => strategy && onStrategyChange(strategy)}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {availableStrategies.map((strategy) => (
              <SelectItem key={strategy} value={strategy}>
                <div className="flex flex-col gap-0.5 py-1">
                  <span className="font-mono text-sm font-medium">{strategy}</span>
                  {routingStrategyDescriptions[strategy] && (
                    <span className="text-xs font-normal text-muted-foreground">
                      {routingStrategyDescriptions[strategy]}
                    </span>
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </ConfigOwnedField>
    </div>
  );
};

export default RoutingStrategySelector;
