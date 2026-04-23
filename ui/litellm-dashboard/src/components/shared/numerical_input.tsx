import React from "react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

interface NumericalInputProps {
  step?: number;
  style?: React.CSSProperties;
  placeholder?: string;
  min?: number;
  max?: number;
  // Callers pass either an onChange(numberOrNull) signature (the @tremor
  // contract this component used to expose) or a React.ChangeEventHandler.
  // We accept both at the signature boundary and bridge to number|null|undefined
  // inside the component.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onChange?: any;
  value?: number | string | null | undefined;
  defaultValue?: number | string | null | undefined;
  name?: string;
  className?: string;
  disabled?: boolean;
  precision?: number;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [key: string]: any;
}

/**
 * Numerical input. Phase-1 shadcn-migrated replacement for the former
 * @tremor NumberInput. Accepts the same prop surface so existing callers
 * continue to work:
 *   - onChange({number | null | undefined}) \u2014 tremor-style
 *   - value / defaultValue as number | string | null | undefined
 *   - step / min / max / precision / placeholder / style / className / disabled
 *
 * `precision` is accepted for API-compat but not enforced (HTML input
 * type=number does not clamp decimal precision; caller code typically
 * rounds before persisting).
 */
const NumericalInput: React.FC<NumericalInputProps> = ({
  step = 0.01,
  style = { width: "100%" },
  placeholder = "Enter a numerical value",
  min,
  max,
  onChange,
  value,
  defaultValue,
  className,
  disabled,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  precision,
  ...rest
}) => {
  const toValue = (v: number | string | null | undefined): string => {
    if (v === null || v === undefined) return "";
    return String(v);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!onChange) return;
    const raw = e.target.value;
    if (raw === "") {
      onChange(null);
      return;
    }
    const num = Number(raw);
    if (Number.isNaN(num)) {
      onChange(null);
      return;
    }
    onChange(num);
  };

  return (
    <Input
      type="number"
      onWheel={(event) => (event.currentTarget as HTMLInputElement).blur()}
      step={step}
      style={style}
      placeholder={placeholder}
      min={min}
      max={max}
      value={value !== undefined ? toValue(value) : undefined}
      defaultValue={
        defaultValue !== undefined ? toValue(defaultValue) : undefined
      }
      onChange={handleChange}
      disabled={disabled}
      className={cn(className)}
      {...rest}
    />
  );
};

export default NumericalInput;
