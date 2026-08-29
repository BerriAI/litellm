import * as React from "react";

import { Input } from "@/components/ui/input";

type NumberInputProps = Omit<React.ComponentProps<"input">, "type" | "value" | "onChange"> & {
  value: number;
  onValueChange: (value: number | null) => void;
};

/**
 * Number input whose field can be emptied while editing: the parent's value is only re-displayed on blur, so
 * backspacing the last digit leaves the field empty instead of snapping straight back to the current value.
 */
const NumberInput = React.forwardRef<HTMLInputElement, NumberInputProps>(
  ({ value, onValueChange, onBlur, ...props }, ref) => {
    const [draft, setDraft] = React.useState<string | null>(null);

    return (
      <Input
        {...props}
        ref={ref}
        type="number"
        value={draft ?? String(value)}
        onChange={(event) => {
          setDraft(event.target.value);
          onValueChange(Number.isNaN(event.target.valueAsNumber) ? null : event.target.valueAsNumber);
        }}
        onBlur={(event) => {
          setDraft(null);
          onBlur?.(event);
        }}
      />
    );
  },
);
NumberInput.displayName = "NumberInput";

export { NumberInput };
