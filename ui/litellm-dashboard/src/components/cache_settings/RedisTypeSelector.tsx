import React from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

interface RedisTypeSelectorProps {
  redisType: string;
  redisTypeDescriptions: { [key: string]: string };
  onTypeChange: (type: string) => void;
}

const RedisTypeSelector: React.FC<RedisTypeSelectorProps> = ({
  redisType,
  redisTypeDescriptions,
  onTypeChange,
}) => {
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">Redis Type</Label>
      <Select value={redisType} onValueChange={onTypeChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="node">Node (Single Instance)</SelectItem>
          <SelectItem value="cluster">Cluster</SelectItem>
          <SelectItem value="sentinel">Sentinel</SelectItem>
          <SelectItem value="semantic">Semantic</SelectItem>
        </SelectContent>
      </Select>
      <p className="text-xs text-muted-foreground">
        {redisTypeDescriptions[redisType] ||
          "Select the type of Redis deployment you're using"}
      </p>
    </div>
  );
};

export default RedisTypeSelector;
