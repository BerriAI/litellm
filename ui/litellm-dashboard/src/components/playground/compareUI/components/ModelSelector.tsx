import React, { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  models: string[];
  loading?: boolean;
  disabled?: boolean;
}

export function ModelSelector({
  value,
  onChange,
  models,
  loading,
  disabled,
}: ModelSelectorProps) {
  const [isAddingCustom, setIsAddingCustom] = useState(false);
  const [customValue, setCustomValue] = useState("");

  const options = useMemo(() => Array.from(new Set(models)).sort(), [models]);
  const displayOptions = useMemo(() => {
    if (value && !options.includes(value)) {
      return [value, ...options];
    }
    return options;
  }, [options, value]);

  const selectValue = isAddingCustom ? "__custom__" : value || undefined;

  const handleSelectChange = (selected: string) => {
    if (selected === "__custom__") {
      setIsAddingCustom(true);
      if (value && !options.includes(value)) {
        setCustomValue(value);
      } else {
        setCustomValue("");
      }
      return;
    }
    setIsAddingCustom(false);
    setCustomValue("");
    onChange(selected);
  };

  const commitCustomValue = () => {
    const trimmed = customValue.trim();
    if (!trimmed) {
      setIsAddingCustom(false);
      setCustomValue("");
      return;
    }
    onChange(trimmed);
    setIsAddingCustom(false);
    setCustomValue("");
  };

  return (
    <div className="flex-1 min-w-0">
      <Select
        value={selectValue}
        onValueChange={handleSelectChange}
        disabled={disabled || loading}
      >
        <SelectTrigger className="w-full">
          <SelectValue
            placeholder={loading ? "Loading models..." : "Select a model"}
          />
        </SelectTrigger>
        <SelectContent>
          {displayOptions.map((model) => (
            <SelectItem key={model} value={model}>
              {model}
            </SelectItem>
          ))}
          <SelectItem value="__custom__">+ Add custom model</SelectItem>
        </SelectContent>
      </Select>
      {isAddingCustom && (
        <Input
          className="mt-2"
          placeholder="Custom Model Name (Enter to add)"
          value={customValue}
          onChange={(e) => setCustomValue(e.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitCustomValue();
            }
          }}
          onBlur={commitCustomValue}
          autoFocus
        />
      )}
    </div>
  );
}
