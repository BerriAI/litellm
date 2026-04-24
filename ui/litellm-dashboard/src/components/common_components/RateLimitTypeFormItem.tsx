import React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface RateLimitTypeFormItemProps {
  /** The type of rate limit - either 'tpm' or 'rpm' */
  type: "tpm" | "rpm";
  /** The form field name */
  name: string;
  /** Whether to show detailed descriptions (default: true) */
  showDetailedDescriptions?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** Initial value for the field */
  initialValue?: string | null;
  /** Antd form instance (optional; for setFieldValue compatibility with
   *  call sites that still use antd forms). */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  form?: any;
  /** Custom onChange handler */
  onChange?: (value: string) => void;
  /** Controlled value (optional). When used with react-hook-form, pass
   *  `value` and `onChange` via a Controller render prop. */
  value?: string;
}

export const RateLimitTypeFormItem: React.FC<RateLimitTypeFormItemProps> = ({
  type,
  name,
  showDetailedDescriptions = true,
  className = "",
  initialValue = null,
  form,
  onChange,
  value,
}) => {
  const limitTypeUpper = type.toUpperCase();
  const limitTypeLower = type.toLowerCase();

  const [internalValue, setInternalValue] = React.useState<string | undefined>(
    value ?? initialValue ?? undefined,
  );

  React.useEffect(() => {
    if (value !== undefined) {
      setInternalValue(value);
    }
  }, [value]);

  const handleChange = (next: string) => {
    setInternalValue(next);
    if (form && typeof form.setFieldValue === "function") {
      form.setFieldValue(name, next);
    }
    if (onChange) {
      onChange(next);
    }
  };

  const tooltipTitle = `Select 'guaranteed_throughput' to prevent overallocating ${limitTypeUpper} limit when the key belongs to a Team with specific ${limitTypeUpper} limits.`;

  return (
    <div className={className}>
      <Label htmlFor={name} className="flex items-center gap-1 mb-2">
        {limitTypeUpper} Rate Limit Type
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="ml-1 h-3 w-3 inline" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{tooltipTitle}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </Label>
      <Select value={internalValue} onValueChange={handleChange}>
        <SelectTrigger id={name} className="w-full">
          <SelectValue placeholder="Select rate limit type" />
        </SelectTrigger>
        <SelectContent>
          {showDetailedDescriptions ? (
            <>
              <SelectItem value="best_effort_throughput">
                <div className="py-1">
                  <div className="font-medium">Default</div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">
                    Best effort throughput - no error if we&apos;re
                    overallocating {limitTypeLower} (Team/Key Limits checked at
                    runtime).
                  </div>
                </div>
              </SelectItem>
              <SelectItem value="guaranteed_throughput">
                <div className="py-1">
                  <div className="font-medium">Guaranteed throughput</div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">
                    Guaranteed throughput - raise an error if we&apos;re
                    overallocating {limitTypeLower} (also checks model-specific
                    limits)
                  </div>
                </div>
              </SelectItem>
              <SelectItem value="dynamic">
                <div className="py-1">
                  <div className="font-medium">Dynamic</div>
                  <div className="text-[11px] text-muted-foreground mt-0.5">
                    If the key has a set {limitTypeUpper} (e.g. 2{" "}
                    {limitTypeUpper}) and there are no 429 errors, it can
                    dynamically exceed the limit when the model being called is
                    not erroring.
                  </div>
                </div>
              </SelectItem>
            </>
          ) : (
            <>
              <SelectItem value="best_effort_throughput">
                Best effort throughput
              </SelectItem>
              <SelectItem value="guaranteed_throughput">
                Guaranteed throughput
              </SelectItem>
              <SelectItem value="dynamic">Dynamic</SelectItem>
            </>
          )}
        </SelectContent>
      </Select>
    </div>
  );
};

export default RateLimitTypeFormItem;
