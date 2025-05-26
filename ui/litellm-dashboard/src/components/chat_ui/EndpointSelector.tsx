import React from "react";
import { Select } from "antd";
import { Text } from "@tremor/react";
import { EndpointType } from "./mode_endpoint_mapping";

interface EndpointSelectorProps {
  endpointType: string; // Accept string to avoid type conflicts
  onEndpointChange: (value: string) => void;
  className?: string;
}

/**
 * A reusable component for selecting API endpoints
 */
const EndpointSelector: React.FC<EndpointSelectorProps> = ({
  endpointType,
  onEndpointChange,
  className,
}) => {
  // Map endpoint types to their display labels
  const endpointOptions = [
    { value: EndpointType.CHAT, label: '/v1/chat/completions' },
    { value: EndpointType.RESPONSES, label: '/v1/responses' },
    { value: EndpointType.IMAGE, label: '/v1/images/generations' },
    { value: EndpointType.IMAGE_EDITS, label: '/v1/images/edits' },
  ];

  return (
    <div className={className}>
      <Text>Endpoint Type:</Text>
      <Select
        value={endpointType}
        style={{ width: "100%" }}
        onChange={onEndpointChange}
        options={endpointOptions}
        className="rounded-md"
      />
    </div>
  );
};

export default EndpointSelector; 