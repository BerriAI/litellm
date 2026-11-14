import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { cx } from "@/lib/cva.config";
import { DEBOUNCE_WAIT_MS } from "@/utils/debounceConstants";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { LucideIcon } from "lucide-react";
import React, { useEffect, useState } from "react";

interface FilterInputProps {
  placeholder?: string;
  value: string;
  onChange: (value: string) => void;
  icon?: LucideIcon;
  className?: string;
  style?: React.CSSProperties;
}

export const FilterInput: React.FC<FilterInputProps> = ({ placeholder, value, onChange, icon: Icon, className }) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const debouncedOnChange = useDebouncedCallback((val: string) => onChange(val), { wait: DEBOUNCE_WAIT_MS });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setLocalValue(newValue);
    debouncedOnChange(newValue);
  };

  return (
    <InputGroup className={cx("w-64", className)}>
      {Icon && (
        <InputGroupAddon>
          <Icon className="size-4 text-muted-foreground" />
        </InputGroupAddon>
      )}
      <InputGroupInput placeholder={placeholder} value={localValue} onChange={handleChange} />
    </InputGroup>
  );
};
