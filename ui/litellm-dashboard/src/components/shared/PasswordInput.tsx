"use client";

import { Eye, EyeOff } from "lucide-react";
import * as React from "react";

import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";

export type PasswordInputProps = Omit<React.ComponentPropsWithoutRef<"input">, "type"> & {
  groupClassName?: string;
};

export const PasswordInput = React.forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ className, groupClassName, disabled, ...props }, ref) => {
    const [revealed, setRevealed] = React.useState(false);

    return (
      <InputGroup className={groupClassName}>
        <InputGroupInput
          {...props}
          ref={ref}
          type={revealed ? "text" : "password"}
          disabled={disabled}
          className={className}
        />
        <InputGroupAddon align="inline-end">
          <InputGroupButton
            size="icon-xs"
            disabled={disabled}
            aria-label={revealed ? "Hide password" : "Show password"}
            onClick={() => setRevealed((current) => !current)}
          >
            {revealed ? <EyeOff /> : <Eye />}
          </InputGroupButton>
        </InputGroupAddon>
      </InputGroup>
    );
  },
);
PasswordInput.displayName = "PasswordInput";
