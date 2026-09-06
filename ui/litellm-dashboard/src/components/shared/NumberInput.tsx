import * as React from "react";

import { Input } from "@/components/ui/input";

type NumberInputProps = Omit<React.ComponentProps<"input">, "type" | "value" | "onChange"> & {
  value: number;
  onValueChange: (value: number) => void;
};

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
          if (!Number.isNaN(event.target.valueAsNumber)) {
            onValueChange(event.target.valueAsNumber);
          }
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
