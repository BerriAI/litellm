import { Select } from "antd";
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
      <Select
        showSearch
        value={endpointType}
        style={{ width: "100%" }}
        onChange={onEndpointChange}
        options={ENDPOINT_OPTIONS}
        className="rounded-md"
        filterOption={(input, option) =>
          (option?.label ?? "").toLowerCase().includes(input.toLowerCase()) ||
          (option?.value ?? "").toLowerCase().includes(input.toLowerCase())
        }
      />
    </div>
  );
};

export default EndpointSelector;
