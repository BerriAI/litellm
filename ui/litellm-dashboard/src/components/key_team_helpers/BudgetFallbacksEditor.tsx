import { Button, Select, Tooltip } from "antd";
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
        <div className="text-xs text-gray-500 mb-2">
          When a model exceeds its per-model budget, requests automatically reroute to fallback models
        </div>
        <Button size="small" onClick={addEntry} icon={<Plus className="w-3 h-3" />}>
          Add Budget Fallback
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="text-xs text-gray-500">
        When a model exceeds its per-model budget, requests automatically reroute to fallback models
      </div>
      {entries.map((entry) => {
        const availablePrimaryOptions = availableModels.filter(
          (m) => m === entry.primaryModel || !usedPrimaryModels.has(m),
        );
        const availableFallbackOptions = availableModels.filter((m) => m !== entry.primaryModel);

        return (
          <div key={entry.id} className="relative rounded-lg border border-gray-200 bg-gray-50 p-4">
            <button
              type="button"
              onClick={() => removeEntry(entry.id)}
              className="absolute top-2 right-2 text-gray-400 hover:text-red-500 transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="mb-3">
              <label className="block text-xs font-medium text-gray-600 mb-1">Primary Model</label>
              <Select
                className="w-full"
                placeholder="Select model"
                value={entry.primaryModel}
                onChange={(v) => {
                  const newFallbacks = entry.fallbackModels.filter((m) => m !== v);
                  updateEntry(entry.id, { primaryModel: v, fallbackModels: newFallbacks });
                }}
                showSearch
                filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
                options={availablePrimaryOptions.map((m) => ({ label: m, value: m }))}
                getPopupContainer={(trigger) => trigger.parentElement || document.body}
              />
            </div>

            <div className="flex items-center justify-center -my-1 mb-2">
              <div className="bg-amber-50 text-amber-600 px-3 py-0.5 rounded-full text-[10px] font-bold border border-amber-100 flex items-center gap-1">
                <ArrowDown className="w-3 h-3" />
                IF BUDGET EXCEEDED, TRY
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-600 mb-1">Fallback Models</label>
              <Select
                mode="multiple"
                className="w-full"
                placeholder={entry.primaryModel ? "Select fallback models" : "Select a primary model first"}
                value={entry.fallbackModels}
                onChange={(values) => updateEntry(entry.id, { fallbackModels: values })}
                disabled={!entry.primaryModel}
                showSearch
                filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
                options={availableFallbackOptions.map((m) => ({ label: m, value: m }))}
                getPopupContainer={(trigger) => trigger.parentElement || document.body}
                maxTagCount="responsive"
                maxTagPlaceholder={(omittedValues) => (
                  <Tooltip
                    styles={{ root: { pointerEvents: "none" } }}
                    title={omittedValues.map(({ value: v }) => v).join(", ")}
                  >
                    <span>+{omittedValues.length} more</span>
                  </Tooltip>
                )}
              />
              {entry.fallbackModels.length > 1 && (
                <div className="text-[10px] text-gray-400 mt-1 ml-1">
                  Tried in order; first model still within its own budget is used
                </div>
              )}
            </div>
          </div>
        );
      })}
      <Button size="small" onClick={addEntry} icon={<Plus className="w-3 h-3" />}>
        Add Budget Fallback
      </Button>
    </div>
  );
}
