import React, { useState, useCallback } from "react";
import { Table, Select, InputNumber, Button, Radio } from "antd";
import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import { PricingCalculatorProps, ModelEntry } from "./types";
import MultiCostResults from "./multi_cost_results";
import { useMultiCostEstimate } from "./use_multi_cost_estimate";

type TimePeriod = "day" | "month";

const generateId = () => `entry-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

const createDefaultEntry = (): ModelEntry => ({
  id: generateId(),
  model: "",
  input_tokens: 1000,
  output_tokens: 500,
  num_requests_per_day: undefined,
  num_requests_per_month: undefined,
});

const PricingCalculator: React.FC<PricingCalculatorProps> = ({
  accessToken,
  models,
}) => {
  const [entries, setEntries] = useState<ModelEntry[]>([createDefaultEntry()]);
  const [timePeriod, setTimePeriod] = useState<TimePeriod>("month");
  const { debouncedFetchForEntry, removeEntry, getMultiModelResult } =
    useMultiCostEstimate(accessToken);

  const handleEntryChange = useCallback(
    (id: string, field: keyof ModelEntry, value: string | number | undefined) => {
      setEntries((prev) => {
        const updated = prev.map((entry) =>
          entry.id === id ? { ...entry, [field]: value } : entry
        );
        const changedEntry = updated.find((e) => e.id === id);
        if (changedEntry && changedEntry.model) {
          debouncedFetchForEntry(changedEntry);
        }
        return updated;
      });
    },
    [debouncedFetchForEntry]
  );

  const handleTimePeriodChange = useCallback((period: TimePeriod) => {
    setTimePeriod(period);
    // Clear the opposite field for all entries when switching
    setEntries((prev) =>
      prev.map((entry) => ({
        ...entry,
        num_requests_per_day: period === "day" ? entry.num_requests_per_day : undefined,
        num_requests_per_month: period === "month" ? entry.num_requests_per_month : undefined,
      }))
    );
  }, []);

  const handleAddEntry = useCallback(() => {
    setEntries((prev) => [...prev, createDefaultEntry()]);
  }, []);

  const handleRemoveEntry = useCallback(
    (id: string) => {
      setEntries((prev) => prev.filter((entry) => entry.id !== id));
      removeEntry(id);
    },
    [removeEntry]
  );

  const multiModelResult = getMultiModelResult(entries);

  const columns = [
    {
      title: "Model",
      dataIndex: "model",
      key: "model",
      width: "35%",
      render: (_: string, record: ModelEntry) => (
        <Select
          showSearch
          placeholder="Select a model"
          value={record.model || undefined}
          onChange={(value) => handleEntryChange(record.id, "model", value)}
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
      ),
    },
    {
      title: "Input Tokens",
      dataIndex: "input_tokens",
      key: "input_tokens",
      width: "18%",
      render: (_: number, record: ModelEntry) => (
        <InputNumber
          min={0}
          value={record.input_tokens}
          onChange={(value) => handleEntryChange(record.id, "input_tokens", value ?? 0)}
          style={{ width: "100%" }}
          size="small"
          formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
        />
      ),
    },
    {
      title: "Output Tokens",
      dataIndex: "output_tokens",
      key: "output_tokens",
      width: "18%",
      render: (_: number, record: ModelEntry) => (
        <InputNumber
          min={0}
          value={record.output_tokens}
          onChange={(value) => handleEntryChange(record.id, "output_tokens", value ?? 0)}
          style={{ width: "100%" }}
          size="small"
          formatter={(value) => `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",")}
        />
      ),
    },
    {
      title: `Requests/${timePeriod === "day" ? "Day" : "Month"}`,
      dataIndex: timePeriod === "day" ? "num_requests_per_day" : "num_requests_per_month",
      key: "num_requests",
      width: "20%",
      render: (_: number | undefined, record: ModelEntry) => (
        <InputNumber
          min={0}
          value={timePeriod === "day" ? record.num_requests_per_day : record.num_requests_per_month}
          onChange={(value) =>
            handleEntryChange(
              record.id,
              timePeriod === "day" ? "num_requests_per_day" : "num_requests_per_month",
              value ?? undefined
            )
          }
          style={{ width: "100%" }}
          size="small"
          placeholder="-"
          formatter={(value) => (value ? `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",") : "")}
        />
      ),
    },
    {
      title: "",
      key: "actions",
      width: 50,
      render: (_: unknown, record: ModelEntry) => (
        <Button
          type="text"
          icon={<DeleteOutlined />}
          onClick={() => handleRemoveEntry(record.id)}
          disabled={entries.length === 1}
          danger
          size="small"
        />
      ),
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end mb-2">
        <Radio.Group
          value={timePeriod}
          onChange={(e) => handleTimePeriodChange(e.target.value)}
          size="small"
          optionType="button"
          buttonStyle="solid"
        >
          <Radio.Button value="day">Per Day</Radio.Button>
          <Radio.Button value="month">Per Month</Radio.Button>
        </Radio.Group>
      </div>

      <Table
        columns={columns}
        dataSource={entries}
        rowKey="id"
        pagination={false}
        size="small"
        footer={() => (
          <Button
            type="dashed"
            onClick={handleAddEntry}
            icon={<PlusOutlined />}
            className="w-full"
          >
            Add Another Model
          </Button>
        )}
      />

      <MultiCostResults multiResult={multiModelResult} timePeriod={timePeriod} />
    </div>
  );
};

export default PricingCalculator;
