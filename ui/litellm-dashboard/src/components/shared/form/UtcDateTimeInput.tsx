"use client";

import dayjs, { type Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";
import * as React from "react";

import { Input } from "@/components/ui/input";

dayjs.extend(utc);

const INPUT_FORMAT = "YYYY-MM-DDTHH:mm:ss";

export interface UtcDateTimeInputProps
  extends Omit<React.ComponentProps<typeof Input>, "value" | "onChange" | "type" | "step" | "ref"> {
  value: Dayjs | null | undefined;
  onChange: (value: Dayjs | null) => void;
}

/**
 * Datetime field whose value is a UTC-mode Dayjs, so the wall clock the operator reads and types is
 * the instant that gets stored: no browser zone ever enters the round trip, and the hour a DST
 * spring-forward skips locally stays typeable.
 */
export const UtcDateTimeInput = React.forwardRef<HTMLInputElement, UtcDateTimeInputProps>(
  ({ value, onChange, ...props }, ref) => (
    <Input
      {...props}
      ref={ref}
      type="datetime-local"
      step={1}
      value={value ? value.format(INPUT_FORMAT) : ""}
      onChange={(event) => onChange(event.target.value ? dayjs.utc(event.target.value) : null)}
    />
  ),
);
UtcDateTimeInput.displayName = "UtcDateTimeInput";
