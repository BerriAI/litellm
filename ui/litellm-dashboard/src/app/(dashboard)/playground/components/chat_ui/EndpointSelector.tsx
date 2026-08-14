import { SearchSelect } from "@/components/shared/SearchSelect";
import React from "react";
import { ENDPOINT_OPTIONS } from "./chatConstants";

interface EndpointSelectorProps {
  endpointType: string; // Accept string to avoid type conflicts
  onEndpointChange: (value: string) => void;
  className?: string;
}

const EndpointSelector: React.FC<EndpointSelectorProps> = ({ endpointType, onEndpointChange, className }) => {
  return (
    <div className={className}>
      <SearchSelect
        value={endpointType}
        onValueChange={onEndpointChange}
        options={ENDPOINT_OPTIONS}
        placeholder="Select an endpoint"
      />
    </div>
  );
};

export default EndpointSelector;
