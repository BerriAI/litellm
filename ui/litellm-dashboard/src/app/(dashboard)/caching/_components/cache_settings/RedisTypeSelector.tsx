import React from "react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CONFIG_SOURCED_DESCRIPTION } from "./CacheFormField";

interface RedisTypeSelectorProps {
  redisType: string;
  redisTypeDescriptions: Readonly<Record<string, string>>;
  onTypeChange: (type: string) => void;
  unavailableTypes?: ReadonlySet<string>;
}

const REDIS_TYPE_LABELS: Readonly<Record<string, string>> = {
  node: "Node (Single Instance)",
  cluster: "Cluster",
  sentinel: "Sentinel",
  semantic: "Semantic",
};

const RedisTypeSelector: React.FC<RedisTypeSelectorProps> = ({
  redisType,
  redisTypeDescriptions,
  onTypeChange,
  unavailableTypes = new Set<string>(),
}) => {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium" htmlFor="redis-type-select">
        Redis Type
      </label>
      <Select value={redisType} onValueChange={(value) => value !== null && onTypeChange(value)}>
        <SelectTrigger id="redis-type-select" className="w-full">
          <SelectValue>{REDIS_TYPE_LABELS[redisType] ?? redisType}</SelectValue>
        </SelectTrigger>
        <SelectContent>
          {Object.entries(REDIS_TYPE_LABELS).map(([value, label]) => (
            <SelectItem key={value} value={value} disabled={unavailableTypes.has(value)}>
              {label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {unavailableTypes.size > 0
          ? CONFIG_SOURCED_DESCRIPTION
          : redisTypeDescriptions[redisType] || "Select the type of Redis deployment you're using"}
      </p>
    </div>
  );
};

export default RedisTypeSelector;
