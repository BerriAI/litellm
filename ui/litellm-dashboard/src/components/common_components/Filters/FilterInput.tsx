import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import debounce from "lodash/debounce";
import { LucideIcon } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";

interface FilterInputProps {
  placeholder?: string;
  value: string;
  onChange: (value: string) => void;
  icon?: LucideIcon;
  className?: string;
  style?: React.CSSProperties;
}

const DEBOUNCE_DELAY = 300;

export const FilterInput: React.FC<FilterInputProps> = ({
  placeholder,
  value,
  onChange,
  icon: Icon,
  className,
}) => {
  const [localValue, setLocalValue] = useState(value);

  useEffect(() => {
    setLocalValue(value);
  }, [value]);

  const debouncedOnChange = useMemo(
    () => debounce((val: string) => onChange(val), DEBOUNCE_DELAY),
    [onChange],
  );

  useEffect(() => {
    return () => {
      debouncedOnChange.cancel();
    };
  }, [debouncedOnChange]);

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const newValue = e.target.value;
      setLocalValue(newValue);
      debouncedOnChange(newValue);
    },
    [debouncedOnChange],
  );

  return (
    <div className={cn("relative w-64", className)}>
      {Icon && (
        <Icon
          size={16}
          className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none"
        />
      )}
      <Input
        placeholder={placeholder}
        value={localValue}
        onChange={handleChange}
        className={cn(Icon && "pl-8")}
      />
    </div>
  );
};
