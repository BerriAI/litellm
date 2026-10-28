import React from "react";
import { CONFIG_OWNED_HINT, ConfigOwnedBadge } from "./ConfigOwned";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface RoutingStrategySelectorProps {
  selectedStrategy: string | null;
  availableStrategies: string[];
  routingStrategyDescriptions: { [key: string]: string };
  routerFieldsMetadata: { [key: string]: any };
  onStrategyChange: (strategy: string) => void;
  ownedByConfig?: boolean;
}

const RoutingStrategySelector: React.FC<RoutingStrategySelectorProps> = ({
  selectedStrategy,
  availableStrategies,
  routingStrategyDescriptions,
  routerFieldsMetadata,
  onStrategyChange,
  ownedByConfig = false,
}) => {
  return (
    <div className="space-y-2 max-w-3xl">
      <div>
        <label className="text-xs font-medium text-foreground uppercase tracking-wide">
          {routerFieldsMetadata["routing_strategy"]?.ui_field_name || "Routing Strategy"}
        </label>
        {ownedByConfig && <ConfigOwnedBadge />}
        <p className="text-xs text-muted-foreground mt-0.5 mb-2">
          {routerFieldsMetadata["routing_strategy"]?.field_description || ""}
        </p>
      </div>
      <div className="routing-strategy-select max-w-3xl">
        <Select
          value={selectedStrategy}
          onValueChange={(strategy: string | null) => strategy && onStrategyChange(strategy)}
          disabled={ownedByConfig}
        >
          <SelectTrigger className="w-full" aria-label={ownedByConfig ? CONFIG_OWNED_HINT : undefined}>
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
      </div>
    </div>
  );
};

export default RoutingStrategySelector;
