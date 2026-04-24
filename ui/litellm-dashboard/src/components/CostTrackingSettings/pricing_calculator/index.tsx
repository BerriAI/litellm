import React, { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { Plus, Trash2 } from "lucide-react";
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

const formatIntegerWithCommas = (value: number | undefined): string => {
  if (value === undefined || value === null || Number.isNaN(value)) return "";
  return `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
};

const parseIntegerInput = (raw: string): number | undefined => {
  const digits = raw.replace(/[^0-9]/g, "");
  if (digits === "") return undefined;
  const parsed = parseInt(digits, 10);
  return Number.isNaN(parsed) ? undefined : parsed;
};

interface NumericCellProps {
  value: number | undefined;
  min?: number;
  placeholder?: string;
  onChange: (value: number | undefined) => void;
  ariaLabel: string;
}

const NumericCell: React.FC<NumericCellProps> = ({ value, min = 0, placeholder, onChange, ariaLabel }) => {
  const [draft, setDraft] = useState<string>(formatIntegerWithCommas(value));

  React.useEffect(() => {
    setDraft(formatIntegerWithCommas(value));
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    const parsed = parseIntegerInput(raw);
    if (parsed !== undefined && parsed < min) {
      setDraft(formatIntegerWithCommas(min));
      onChange(min);
      return;
    }
    setDraft(parsed === undefined ? "" : formatIntegerWithCommas(parsed));
    onChange(parsed);
  };

  return (
    <Input
      aria-label={ariaLabel}
      value={draft}
      placeholder={placeholder}
      inputMode="numeric"
      onChange={handleChange}
      className="h-8 text-sm"
    />
  );
};

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
  const requestsHeader = `Requests/${timePeriod === "day" ? "Day" : "Month"}`;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end mb-2">
        <div className="inline-flex rounded-md border border-border overflow-hidden text-xs">
          <button
            type="button"
            onClick={() => handleTimePeriodChange("day")}
            className={cn(
              "px-3 py-1",
              timePeriod === "day"
                ? "bg-primary text-primary-foreground"
                : "bg-background text-foreground hover:bg-muted",
            )}
          >
            Per Day
          </button>
          <button
            type="button"
            onClick={() => handleTimePeriodChange("month")}
            className={cn(
              "px-3 py-1 border-l border-border",
              timePeriod === "month"
                ? "bg-primary text-primary-foreground"
                : "bg-background text-foreground hover:bg-muted",
            )}
          >
            Per Month
          </button>
        </div>
      </div>

      <div className="rounded-md border border-border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-[35%] h-10">Model</TableHead>
              <TableHead className="w-[18%] h-10">Input Tokens</TableHead>
              <TableHead className="w-[18%] h-10">Output Tokens</TableHead>
              <TableHead className="w-[20%] h-10">{requestsHeader}</TableHead>
              <TableHead className="w-[50px] h-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {entries.map((entry) => {
              const requestsValue = timePeriod === "day" ? entry.num_requests_per_day : entry.num_requests_per_month;
              const requestsField = timePeriod === "day" ? "num_requests_per_day" : "num_requests_per_month";
              return (
                <TableRow key={entry.id}>
                  <TableCell className="py-2 align-middle">
                    <Select
                      value={entry.model || undefined}
                      onValueChange={(value) => handleEntryChange(entry.id, "model", value)}
                    >
                      <SelectTrigger className="h-8 text-sm">
                        <SelectValue placeholder="Select a model" />
                      </SelectTrigger>
                      <SelectContent>
                        {models.map((model) => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell className="py-2 align-middle">
                    <NumericCell
                      value={entry.input_tokens}
                      onChange={(value) => handleEntryChange(entry.id, "input_tokens", value ?? 0)}
                      ariaLabel="Input Tokens"
                    />
                  </TableCell>
                  <TableCell className="py-2 align-middle">
                    <NumericCell
                      value={entry.output_tokens}
                      onChange={(value) => handleEntryChange(entry.id, "output_tokens", value ?? 0)}
                      ariaLabel="Output Tokens"
                    />
                  </TableCell>
                  <TableCell className="py-2 align-middle">
                    <NumericCell
                      value={requestsValue}
                      placeholder="-"
                      onChange={(value) => handleEntryChange(entry.id, requestsField, value)}
                      ariaLabel={requestsHeader}
                    />
                  </TableCell>
                  <TableCell className="py-2 align-middle">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-destructive hover:text-destructive"
                      onClick={() => handleRemoveEntry(entry.id)}
                      disabled={entries.length === 1}
                      aria-label="Remove"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
        <div className="border-t border-border p-2">
          <Button variant="outline" onClick={handleAddEntry} className="w-full border-dashed">
            <Plus className="h-4 w-4" />
            Add Another Model
          </Button>
        </div>
      </div>

      <MultiCostResults multiResult={multiModelResult} timePeriod={timePeriod} />
    </div>
  );
};

export default PricingCalculator;
