import React from "react";
import { Select } from "antd";
import { Text } from "@tremor/react";
import { ENDPOINT_OPTIONS } from "./chatConstants";

interface EndpointSelectorProps {
  endpointType: string; // Accept string to avoid type conflicts
  onEndpointChange: (value: string) => void;
  className?: string;
}

const EndpointSelector: React.FC<EndpointSelectorProps> = ({ endpointType, onEndpointChange, className }) => {
  return (
    <div className={className}>
      <Text>Endpoint Type:</Text>
      <Select
        showSearch
        value={endpointType}
        style={{ width: "100%" }}
        onChange={onEndpointChange}
        options={ENDPOINT_OPTIONS}
        className="rounded-md"
      />
    </div>
  );
};

export default EndpointSelector;
