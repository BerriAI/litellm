import React from "react";
import { CircleHelp } from "lucide-react";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

type RateLimitType = "tpm" | "rpm";

interface RateLimitTypeOption {
  value: string;
  label: string;
  description: string;
}

interface RateLimitTypeFormItemProps {
  /** The type of rate limit - either 'tpm' or 'rpm' */
  type: RateLimitType;
  /** The form field name */
  name: string;
  /** Whether to show detailed descriptions (default: true) */
  showDetailedDescriptions?: boolean;
  /** Additional CSS classes */
  className?: string;
  value?: string | null;
  /** Custom onChange handler */
  onChange?: (value: string) => void;
  id?: string;
  disabled?: boolean;
  "aria-invalid"?: true | undefined;
  "aria-describedby"?: string | undefined;
}

const rateLimitTypeOptions = (type: RateLimitType): RateLimitTypeOption[] => {
  const upper = type.toUpperCase();
  const lower = type.toLowerCase();
  return [
    {
      value: "best_effort_throughput",
      label: "Default",
      description: `Best effort throughput - no error if we're overallocating ${lower} (Team/Key Limits checked at runtime).`,
    },
    {
      value: "guaranteed_throughput",
      label: "Guaranteed throughput",
      description: `Guaranteed throughput - raise an error if we're overallocating ${lower} (also checks model-specific limits)`,
    },
    {
      value: "dynamic",
      label: "Dynamic",
      description: `If the key has a set ${upper} (e.g. 2 ${upper}) and there are no 429 errors, it can dynamically exceed the limit when the model being called is not erroring.`,
    },
  ];
};

const plainLabels: Record<string, string> = {
  best_effort_throughput: "Best effort throughput",
  guaranteed_throughput: "Guaranteed throughput",
  dynamic: "Dynamic",
};

const rateLimitTypeLabelText = (type: RateLimitType): string => `${type.toUpperCase()} Rate Limit Type`;

const rateLimitTypeTooltip = (type: RateLimitType): string =>
  `Select 'guaranteed_throughput' to prevent overallocating ${type.toUpperCase()} limit when the key belongs to a Team with specific ${type.toUpperCase()} limits.`;

export const RateLimitTypeFormItem: React.FC<RateLimitTypeFormItemProps> = ({
  type,
  name,
  showDetailedDescriptions = true,
  className = "",
  value,
  onChange,
  id,
  disabled,
  "aria-invalid": ariaInvalid,
  "aria-describedby": ariaDescribedBy,
}) => {
  const controlId = id ?? `rate-limit-type-${name}`;
  const options = rateLimitTypeOptions(type);

  return (
    <div className={className}>
      <TooltipProvider>
        <label htmlFor={controlId} className="mb-2 flex items-center gap-1 text-sm text-foreground">
          {rateLimitTypeLabelText(type)}
          <Tooltip>
            <TooltipTrigger
              render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />}
              aria-label={rateLimitTypeTooltip(type)}
            />
            <TooltipContent>{rateLimitTypeTooltip(type)}</TooltipContent>
          </Tooltip>
        </label>
      </TooltipProvider>
      <Select
        value={value ?? null}
        onValueChange={(next: string | null) => next !== null && onChange?.(next)}
        disabled={disabled}
      >
        <SelectTrigger id={controlId} className="w-full" aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
          <SelectValue placeholder="Select rate limit type">
            {(selected: string | null) =>
              selected === null ? "Select rate limit type" : plainLabels[selected] ?? selected
            }
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {options.map((option) =>
            showDetailedDescriptions ? (
              <SelectItem key={option.value} value={option.value} title={option.label}>
                <span className="flex flex-col py-1">
                  <span className="font-medium">{option.label}</span>
                  <span className="mt-0.5 text-[11px] text-muted-foreground">{option.description}</span>
                </span>
              </SelectItem>
            ) : (
              <SelectItem key={option.value} value={option.value} title={plainLabels[option.value]}>
                {plainLabels[option.value]}
              </SelectItem>
            ),
          )}
        </SelectContent>
      </Select>
    </div>
  );
};

export default RateLimitTypeFormItem;
