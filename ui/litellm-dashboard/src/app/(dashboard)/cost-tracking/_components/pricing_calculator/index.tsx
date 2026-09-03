import React, { useState, useCallback } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableFooter, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { SearchSelect } from "@/components/shared/SearchSelect";
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

const PricingCalculator: React.FC<PricingCalculatorProps> = ({ accessToken, models }) => {
  const [entries, setEntries] = useState<ModelEntry[]>([createDefaultEntry()]);
  const [timePeriod, setTimePeriod] = useState<TimePeriod>("month");
  const { debouncedFetchForEntry, removeEntry, getMultiModelResult } = useMultiCostEstimate(accessToken);

  const handleEntryChange = useCallback(
    (id: string, field: keyof ModelEntry, value: string | number | undefined) => {
      setEntries((prev) => {
        const updated = prev.map((entry) => (entry.id === id ? { ...entry, [field]: value } : entry));
        const changedEntry = updated.find((e) => e.id === id);
        if (changedEntry && changedEntry.model) {
          debouncedFetchForEntry(changedEntry);
        }
        return updated;
      });
    },
    [debouncedFetchForEntry],
  );

  const handleTimePeriodChange = useCallback((period: TimePeriod) => {
    setTimePeriod(period);
    // Clear the opposite field for all entries when switching
    setEntries((prev) =>
      prev.map((entry) => ({
        ...entry,
        num_requests_per_day: period === "day" ? entry.num_requests_per_day : undefined,
        num_requests_per_month: period === "month" ? entry.num_requests_per_month : undefined,
      })),
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
    [removeEntry],
  );

  const multiModelResult = getMultiModelResult(entries);

  const modelOptions = models.map((model) => ({ label: model, value: model }));
  const requestsField = timePeriod === "day" ? "num_requests_per_day" : "num_requests_per_month";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end mb-2">
        <RadioGroup
          value={timePeriod}
          onValueChange={(value) => handleTimePeriodChange(value as TimePeriod)}
          className="flex w-auto items-center gap-4"
        >
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <RadioGroupItem value="day" />
            Per Day
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <RadioGroupItem value="month" />
            Per Month
          </label>
        </RadioGroup>
      </div>

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-[35%]">Model</TableHead>
            <TableHead className="w-[18%]">Input Tokens</TableHead>
            <TableHead className="w-[18%]">Output Tokens</TableHead>
            <TableHead className="w-[20%]">Requests/{timePeriod === "day" ? "Day" : "Month"}</TableHead>
            <TableHead className="w-[50px]">
              <span className="sr-only">Actions</span>
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {entries.map((record, index) => (
            <TableRow key={record.id}>
              <TableCell className="whitespace-normal">
                <SearchSelect
                  options={modelOptions}
                  value={record.model || undefined}
                  onValueChange={(value) => handleEntryChange(record.id, "model", value)}
                  placeholder="Select a model"
                />
              </TableCell>
              <TableCell>
                <Input
                  type="number"
                  min={0}
                  className="h-8"
                  value={record.input_tokens}
                  onChange={(e) =>
                    handleEntryChange(record.id, "input_tokens", e.target.value === "" ? 0 : Number(e.target.value))
                  }
                />
              </TableCell>
              <TableCell>
                <Input
                  type="number"
                  min={0}
                  className="h-8"
                  value={record.output_tokens}
                  onChange={(e) =>
                    handleEntryChange(record.id, "output_tokens", e.target.value === "" ? 0 : Number(e.target.value))
                  }
                />
              </TableCell>
              <TableCell>
                <Input
                  type="number"
                  min={0}
                  className="h-8"
                  placeholder="-"
                  value={record[requestsField] ?? ""}
                  onChange={(e) =>
                    handleEntryChange(
                      record.id,
                      requestsField,
                      e.target.value === "" ? undefined : Number(e.target.value),
                    )
                  }
                />
              </TableCell>
              <TableCell>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  aria-label={`Remove model row ${index + 1}`}
                  onClick={() => handleRemoveEntry(record.id)}
                  disabled={entries.length === 1}
                  className="text-destructive"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
        <TableFooter>
          <TableRow>
            <TableCell colSpan={5}>
              <Button variant="outline" onClick={handleAddEntry} className="w-full border-dashed">
                <Plus className="size-3.5" />
                Add Another Model
              </Button>
            </TableCell>
          </TableRow>
        </TableFooter>
      </Table>

      <MultiCostResults multiResult={multiModelResult} timePeriod={timePeriod} />
    </div>
  );
};

export default PricingCalculator;
