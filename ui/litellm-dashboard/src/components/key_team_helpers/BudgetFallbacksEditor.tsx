import { MultiSelect } from "@/components/shared/MultiSelect";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { Button } from "@/components/ui/button";
import { ArrowDown, Plus, X } from "lucide-react";
import React, { useState } from "react";

interface FallbackEntry {
  id: string;
  primaryModel: string | null;
  fallbackModels: string[];
}

interface BudgetFallbacksEditorProps {
  value: Record<string, string[]>;
  onChange: (v: Record<string, string[]>) => void;
  availableModels: string[];
}

const entriesToDict = (entries: readonly FallbackEntry[]): Record<string, string[]> =>
  Object.fromEntries(
    entries
      .filter(
        (e): e is FallbackEntry & { primaryModel: string } => e.primaryModel !== null && e.fallbackModels.length > 0,
      )
      .map((e) => [e.primaryModel, e.fallbackModels]),
  );

const dictToEntries = (dict: Record<string, string[]>): FallbackEntry[] => {
  const keys = Object.keys(dict);
  if (keys.length === 0) return [];
  return keys.map((model, i) => ({
    id: String(i + 1),
    primaryModel: model,
    fallbackModels: dict[model],
  }));
};

export function BudgetFallbacksEditor({ value, onChange, availableModels }: BudgetFallbacksEditorProps) {
  const [entries, setEntries] = useState<FallbackEntry[]>(() => dictToEntries(value));

  const emitChange = (updated: FallbackEntry[]) => {
    setEntries(updated);
    onChange(entriesToDict(updated));
  };

  const addEntry = () => {
    emitChange([...entries, { id: Date.now().toString(), primaryModel: null, fallbackModels: [] }]);
  };

  const removeEntry = (id: string) => {
    emitChange(entries.filter((e) => e.id !== id));
  };

  const updateEntry = (id: string, patch: Partial<FallbackEntry>) => {
    emitChange(entries.map((e) => (e.id === id ? { ...e, ...patch } : e)));
  };

  const usedPrimaryModels = new Set(entries.map((e) => e.primaryModel).filter(Boolean));

  if (entries.length === 0) {
    return (
      <div>
        <div className="text-xs text-muted-foreground mb-2">
          When a model exceeds its per-model budget, requests automatically reroute to fallback models
        </div>
        <Button variant="outline" size="sm" onClick={addEntry}>
          <Plus className="w-3 h-3" />
          Add Budget Fallback
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-xs text-muted-foreground">
        When a model exceeds its per-model budget, requests automatically reroute to fallback models
      </div>
      {entries.map((entry) => {
        const availablePrimaryOptions = availableModels.filter(
          (m) => m === entry.primaryModel || !usedPrimaryModels.has(m),
        );
        const availableFallbackOptions = availableModels.filter((m) => m !== entry.primaryModel);

        return (
          <div key={entry.id} className="relative rounded-lg border border-border bg-muted p-4">
            <button
              type="button"
              onClick={() => removeEntry(entry.id)}
              className="absolute top-2 right-2 text-muted-foreground hover:text-destructive transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="mb-3">
              <label className="block text-xs font-medium text-muted-foreground mb-1">Primary Model</label>
              <SearchSelect
                options={availablePrimaryOptions.map((m) => ({ label: m, value: m }))}
                value={entry.primaryModel ?? ""}
                onValueChange={(v) => {
                  const newFallbacks = entry.fallbackModels.filter((m) => m !== v);
                  updateEntry(entry.id, { primaryModel: v === "" ? null : v, fallbackModels: newFallbacks });
                }}
                placeholder="Select model"
                emptyText="No models found"
              />
            </div>

            <div className="flex items-center justify-center -my-1 mb-2">
              <div className="bg-warning/10 text-warning px-3 py-0.5 rounded-full text-[10px] font-bold border border-warning/15 flex items-center gap-1">
                <ArrowDown className="w-3 h-3" />
                IF BUDGET EXCEEDED, TRY
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-muted-foreground mb-1">Fallback Models</label>
              <MultiSelect
                options={availableFallbackOptions.map((m) => ({ label: m, value: m }))}
                value={entry.fallbackModels}
                onValueChange={(values) => updateEntry(entry.id, { fallbackModels: values })}
                placeholder={entry.primaryModel ? "Select fallback models" : "Select a primary model first"}
                emptyText="No models found"
                disabled={!entry.primaryModel}
                className="w-full"
              />
              {entry.fallbackModels.length > 1 && (
                <div className="text-[10px] text-muted-foreground mt-1 ml-1">
                  Tried in order; first model still within its own budget is used
                </div>
              )}
            </div>
          </div>
        );
      })}
      <Button variant="outline" size="sm" onClick={addEntry}>
        <Plus className="w-3 h-3" />
        Add Budget Fallback
      </Button>
    </div>
  );
}
