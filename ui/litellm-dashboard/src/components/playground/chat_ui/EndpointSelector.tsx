import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import React from "react";
import { ENDPOINT_OPTIONS } from "./chatConstants";

interface EndpointSelectorProps {
  endpointType: string;
  onEndpointChange: (value: string) => void;
  className?: string;
}

const EndpointSelector: React.FC<EndpointSelectorProps> = ({
  endpointType,
  onEndpointChange,
  className,
}) => {
  return (
    <div className={cn(className)}>
      <Select value={endpointType} onValueChange={onEndpointChange}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {ENDPOINT_OPTIONS.map((opt) => (
            <SelectItem
              key={opt.value as string}
              value={opt.value as string}
            >
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export default EndpointSelector;
