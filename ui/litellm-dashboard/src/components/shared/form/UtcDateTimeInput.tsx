"use client";

import * as React from "react";
import dayjs, { type Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";

import { Input } from "@/components/ui/input";
import { cn } from "@/lib/cva.config";

dayjs.extend(utc);

const MINUTE_FORMAT = "YYYY-MM-DDTHH:mm";
const SECOND_FORMAT = "YYYY-MM-DDTHH:mm:ss";

export interface UtcDateTimeInputProps
  extends Omit<React.ComponentProps<"input">, "value" | "onChange" | "type" | "step"> {
  value: Dayjs | null | undefined;
  onChange: (value: Dayjs | null) => void;
}

/**
 * The wall clock the operator sees is the wall clock that gets stored, so the value stays in UTC
 * mode end to end instead of being converted through the browser zone - see utils/ptuDatetime.
 */
const toInputValue = (value: Dayjs | null | undefined): string => {
  if (!value || typeof value.format !== "function" || !value.isValid()) {
    return "";
  }
  return value.second() === 0 && value.millisecond() === 0 ? value.format(MINUTE_FORMAT) : value.format(SECOND_FORMAT);
};

const toUtcValue = (raw: string): Dayjs | null => {
  if (!raw) {
    return null;
  }
  const parsed = dayjs.utc(raw);
  return parsed.isValid() ? parsed : null;
};

export const UtcDateTimeInput = React.forwardRef<HTMLInputElement, UtcDateTimeInputProps>(
  ({ value, onChange, className, ...props }, ref) => (
    <Input
      {...props}
      ref={ref}
      type="datetime-local"
      step={1}
      className={cn("w-full", className)}
      value={toInputValue(value)}
      onChange={(event) => onChange(toUtcValue(event.target.value))}
    />
  ),
);
UtcDateTimeInput.displayName = "UtcDateTimeInput";
