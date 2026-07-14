import React from "react";
import { Select } from "antd";
import { PlusOutlined, MinusCircleOutlined } from "@ant-design/icons";
import NumericalInput from "./numerical_input";

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
        <div key={index} className="flex items-center mb-4 gap-4">
          <div style={{ width: "180px" }}>
            <span className="text-xs text-gray-500">Type</span>
            <Select
              disabled
              value="message"
              options={[{ value: "message", label: "Message" }]}
              className="w-full"
              data-testid={`cache-control-location-select-${index}`}
            />
          </div>

          <div style={{ width: "180px" }}>
            <span className="text-xs text-gray-500">Role</span>
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

          <div style={{ width: "180px" }}>
            <span className="text-xs text-gray-500">Index</span>
            <NumericalInput
              type="number"
              placeholder="Optional"
              step={1}
              value={point.index}
              onChange={(newIndex: string) =>
                updatePoint(index, { index: newIndex === "" ? undefined : Number(newIndex) })
              }
            />
          </div>

          {points.length > 1 && (
            <MinusCircleOutlined
              className="text-red-500 cursor-pointer text-lg ml-12"
              onClick={() => onChange(points.filter((_, i) => i !== index))}
            />
          )}
        </div>
      ))}

      <button
        type="button"
        className="flex items-center justify-center w-full border border-dashed border-gray-300 py-2 px-4 text-gray-600 hover:text-blue-600 hover:border-blue-300 transition-all rounded"
        onClick={() => onChange([...points, { location: "message" as const }])}
      >
        <PlusOutlined className="mr-2" />
        Add Injection Point
      </button>
    </>
  );
};

export default CacheControlInjectionPointsEditor;
