import React, { useMemo, useState } from "react";
import { Select } from "antd";
import { TextInput } from "@tremor/react";
interface ModelSelectorProps {
  value: string;
  onChange: (value: string) => void;
  models: string[];
  loading?: boolean;
  disabled?: boolean;
}
export function ModelSelector({ value, onChange, models, loading, disabled }: ModelSelectorProps) {
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
      <Select<string>
        value={selectValue}
        onChange={handleSelectChange}
        disabled={disabled}
        loading={loading}
        placeholder={loading ? "Loading models..." : "Select a model"}
        className="w-full rounded-md"
        showSearch
        optionFilterProp="children"
      >
        {displayOptions.map((model) => (
          <Select.Option key={model} value={model}>
            {model}
          </Select.Option>
        ))}
        <Select.Option value="__custom__">+ Add custom model</Select.Option>
      </Select>
      {isAddingCustom && (
        <TextInput
          className="mt-2"
          placeholder="Custom Model Name (Enter to add)"
          value={customValue}
          onValueChange={setCustomValue}
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
