import React from "react";
import { Select, InputNumber, Button, Tooltip } from "antd";
import { DeleteOutlined } from "@ant-design/icons";
import { ModelEntry } from "./types";

interface ModelEntryRowProps {
  entry: ModelEntry;
  models: string[];
  onChange: (id: string, field: keyof ModelEntry, value: string | number | undefined) => void;
  onRemove: (id: string) => void;
  canRemove: boolean;
}

const ModelEntryRow: React.FC<ModelEntryRowProps> = ({
  entry,
  models,
  onChange,
  onRemove,
  canRemove,
}) => {
  return (
    <div className="flex items-start gap-3 p-4 bg-gray-50 rounded-lg border border-gray-200">
      <div className="flex-1 grid grid-cols-5 gap-3">
        <div>
          <label className="text-xs text-gray-500 block mb-1">Model</label>
          <Select
            showSearch
            placeholder="Select model"
            value={entry.model || undefined}
            onChange={(value) => onChange(entry.id, "model", value)}
            optionFilterProp="label"
            filterOption={(input, option) =>
              String(option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
            options={models.map((model) => ({
              value: model,
              label: model,
            }))}
            style={{ width: "100%" }}
            size="small"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Input Tokens</label>
          <InputNumber
            min={0}
            value={entry.input_tokens}
            onChange={(value) => onChange(entry.id, "input_tokens", value ?? 0)}
            style={{ width: "100%" }}
            size="small"
            formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Output Tokens</label>
          <InputNumber
            min={0}
            value={entry.output_tokens}
            onChange={(value) => onChange(entry.id, "output_tokens", value ?? 0)}
            style={{ width: "100%" }}
            size="small"
            formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Requests/Day</label>
          <InputNumber
            min={0}
            value={entry.num_requests_per_day}
            onChange={(value) => onChange(entry.id, "num_requests_per_day", value ?? undefined)}
            style={{ width: "100%" }}
            size="small"
            placeholder="Optional"
            formatter={(value) => value ? `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",") : ""}
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 block mb-1">Requests/Month</label>
          <InputNumber
            min={0}
            value={entry.num_requests_per_month}
            onChange={(value) => onChange(entry.id, "num_requests_per_month", value ?? undefined)}
            style={{ width: "100%" }}
            size="small"
            placeholder="Optional"
            formatter={(value) => value ? `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",") : ""}
          />
        </div>
      </div>
      <div className="pt-5">
        <Tooltip title={canRemove ? "Remove model" : "At least one model required"}>
          <Button
            type="text"
            icon={<DeleteOutlined />}
            onClick={() => onRemove(entry.id)}
            disabled={!canRemove}
            danger
            size="small"
          />
        </Tooltip>
      </div>
    </div>
  );
};

export default ModelEntryRow;

