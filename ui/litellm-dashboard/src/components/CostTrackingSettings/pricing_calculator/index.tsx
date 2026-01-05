import React, { useState, useCallback } from "react";
import { Button, Text } from "@tremor/react";
import { PlusOutlined } from "@ant-design/icons";
import { PricingCalculatorProps, ModelEntry } from "./types";
import ModelEntryRow from "./model_entry_row";
import MultiCostResults from "./multi_cost_results";
import { useMultiCostEstimate } from "./use_multi_cost_estimate";

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

  // Note: Fetching is triggered by handleEntryChange when model is selected

  const multiModelResult = getMultiModelResult(entries);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between mb-2">
        <Text className="text-sm text-gray-600">
          Add models to estimate costs. Each model can have its own token counts and request volumes.
        </Text>
        <Button
          size="xs"
          variant="secondary"
          icon={PlusOutlined}
          onClick={handleAddEntry}
        >
          Add Model
        </Button>
      </div>

      <div className="space-y-3">
        {entries.map((entry) => (
          <ModelEntryRow
            key={entry.id}
            entry={entry}
            models={models}
            onChange={handleEntryChange}
            onRemove={handleRemoveEntry}
            canRemove={entries.length > 1}
          />
        ))}
      </div>

      <MultiCostResults multiResult={multiModelResult} />
    </div>
  );
};

export default PricingCalculator;
