"use client";

import moment from "moment";
import { CalendarDays } from "lucide-react";
import { type ComponentProps, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/cva.config";

import { QUICK_SELECT_OPTIONS } from "./constants";
import { getTimeRangeDisplay } from "./logs_utils";

const DATETIME_LOCAL_FORMAT = "YYYY-MM-DDTHH:mm";

export interface LogsTimeRange {
  startTime: string;
  endTime: string;
  isCustomDate: boolean;
  interval: { value: number; unit: string };
}

const relativeTimeRange = (interval: LogsTimeRange["interval"]): LogsTimeRange => ({
  startTime: moment()
    .subtract(interval.value, interval.unit as moment.unitOfTime.DurationConstructor)
    .format(DATETIME_LOCAL_FORMAT),
  endTime: moment().format(DATETIME_LOCAL_FORMAT),
  isCustomDate: false,
  interval,
});

export const defaultLogsTimeRange = (): LogsTimeRange => relativeTimeRange({ value: 24, unit: "hours" });

export function LogsToolbar({ className, ...props }: ComponentProps<"div">) {
  return <div className={cn("flex flex-wrap items-center gap-2", className)} {...props} />;
}

interface LogsTimeRangePickerProps {
  value: LogsTimeRange;
  onValueChange: (value: LogsTimeRange) => void;
}

export function LogsTimeRangePicker({ value, onValueChange }: LogsTimeRangePickerProps) {
  const [open, setOpen] = useState(false);

  const selectedOption = QUICK_SELECT_OPTIONS.find(
    (option) => option.value === value.interval.value && option.unit === value.interval.unit,
  );
  const displayLabel = value.isCustomDate
    ? getTimeRangeDisplay(true, value.startTime, value.endTime)
    : selectedOption?.label;

  return (
    <>
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger
          render={
            <Button variant="outline" size="sm" className="gap-2">
              <CalendarDays className="size-4" />
              {displayLabel}
            </Button>
          }
        />
        <PopoverContent align="start" className="w-64 p-2">
          <div className="space-y-1">
            {QUICK_SELECT_OPTIONS.map((option) => (
              <Button
                key={option.label}
                variant="ghost"
                className="w-full justify-start font-normal"
                onClick={() => {
                  onValueChange(relativeTimeRange({ value: option.value, unit: option.unit }));
                  setOpen(false);
                }}
              >
                {option.label}
              </Button>
            ))}
            <div className="my-2 border-t" />
            <Button
              variant="ghost"
              className="w-full justify-start font-normal"
              onClick={() => onValueChange({ ...value, isCustomDate: !value.isCustomDate })}
            >
              Custom Range
            </Button>
          </div>
        </PopoverContent>
      </Popover>

      {value.isCustomDate && (
        <div className="flex items-center gap-2">
          <Input
            type="datetime-local"
            className="w-auto"
            value={value.startTime}
            onChange={(event) => onValueChange({ ...value, startTime: event.target.value })}
          />
          <span className="text-sm text-muted-foreground">to</span>
          <Input
            type="datetime-local"
            className="w-auto"
            value={value.endTime}
            onChange={(event) => onValueChange({ ...value, endTime: event.target.value })}
          />
        </div>
      )}
    </>
  );
}

export function LogsToolbarSwitch({ label, ...props }: ComponentProps<typeof Switch> & { label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium">{label}</span>
      <Switch aria-label={label} {...props} />
    </div>
  );
}
