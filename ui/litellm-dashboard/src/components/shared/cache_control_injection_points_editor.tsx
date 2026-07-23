import React from "react";
import { Button, Flex, InputNumber, Select, Typography } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";

const { Text } = Typography;

export interface CacheControlInjectionPoint {
  location: "message";
  role?: "user" | "system" | "assistant";
  index?: number;
}

interface CacheControlInjectionPointsEditorProps {
  value: CacheControlInjectionPoint[];
  onChange: (points: CacheControlInjectionPoint[]) => void;
}

const CacheControlInjectionPointsEditor: React.FC<CacheControlInjectionPointsEditorProps> = ({ value, onChange }) => {
  const points = value.length > 0 ? value : [{ location: "message" as const }];

  const updatePoint = (index: number, patch: Partial<CacheControlInjectionPoint>) => {
    onChange(points.map((point, i) => (i === index ? { ...point, ...patch } : point)));
  };

  return (
    <>
      {points.map((point, index) => (
        <Flex key={index} align="end" gap="middle" wrap className="mb-4">
          <div className="flex-1 min-w-40">
            <Text type="secondary" className="block text-xs mb-1">
              Type
            </Text>
            <Select
              disabled
              value="message"
              options={[{ value: "message", label: "Message" }]}
              className="w-full"
              data-testid={`cache-control-location-select-${index}`}
            />
          </div>

          <div className="flex-1 min-w-40">
            <Text type="secondary" className="block text-xs mb-1">
              Role
            </Text>
            <Select
              placeholder="Select a role"
              allowClear
              value={point.role}
              onChange={(role) => updatePoint(index, { role })}
              options={[
                { value: "user", label: "User" },
                { value: "system", label: "System" },
                { value: "assistant", label: "Assistant" },
              ]}
              className="w-full"
              data-testid={`cache-control-role-select-${index}`}
            />
          </div>

          <div className="flex-1 min-w-40">
            <Text type="secondary" className="block text-xs mb-1">
              Index
            </Text>
            <InputNumber
              placeholder="Optional"
              step={1}
              precision={0}
              value={point.index}
              onChange={(newIndex) => updatePoint(index, { index: newIndex ?? undefined })}
              className="w-full"
              data-testid={`cache-control-index-input-${index}`}
            />
          </div>

          {points.length > 1 && (
            <Button
              type="text"
              danger
              aria-label={`Remove injection point ${index + 1}`}
              icon={<MinusCircleOutlined />}
              onClick={() => onChange(points.filter((_, i) => i !== index))}
            />
          )}
        </Flex>
      ))}

      <Button
        type="dashed"
        block
        icon={<PlusOutlined />}
        onClick={() => onChange([...points, { location: "message" as const }])}
      >
        Add Injection Point
      </Button>
    </>
  );
};

export default CacheControlInjectionPointsEditor;
