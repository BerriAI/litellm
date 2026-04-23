import React from "react";
import { Form } from "antd";
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
  /** Form instance for setting field values */
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  form?: any;
  /** Custom onChange handler */
  onChange?: (value: string) => void;
}

export const RateLimitTypeFormItem: React.FC<RateLimitTypeFormItemProps> = ({
  type,
  name,
  showDetailedDescriptions = true,
  className = "",
  initialValue = null,
  form,
  onChange,
}) => {
  const limitTypeUpper = type.toUpperCase();
  const limitTypeLower = type.toLowerCase();

  const handleChange = (value: string) => {
    if (form) {
      form.setFieldValue(name, value);
    }
    if (onChange) {
      onChange(value);
    }
  };

  const tooltipTitle = `Select 'guaranteed_throughput' to prevent overallocating ${limitTypeUpper} limit when the key belongs to a Team with specific ${limitTypeUpper} limits.`;

  return (
    <Form.Item
      label={
        <span>
          {limitTypeUpper} Rate Limit Type{" "}
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-1 h-3 w-3 inline" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                {tooltipTitle}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </span>
      }
      name={name}
      initialValue={initialValue}
      className={className}
    >
      <Select
        defaultValue={showDetailedDescriptions ? "default" : undefined}
        onValueChange={handleChange}
      >
        <SelectTrigger className="w-full">
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
    </Form.Item>
  );
};

export default RateLimitTypeFormItem;
