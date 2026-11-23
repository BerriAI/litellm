import React from "react";
import type { ParseKeys, TFunction } from "i18next";
import { CircleHelp } from "lucide-react";
import { useTranslation } from "react-i18next";

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

const rateLimitTypeOptions = (type: RateLimitType, t: TFunction): RateLimitTypeOption[] => {
  const upper = type.toUpperCase();
  const lower = type.toLowerCase();
  return [
    {
      value: "best_effort_throughput",
      label: t("rateLimitTypeFormItem.options.default"),
      description: t("rateLimitTypeFormItem.options.defaultDescription", { lower }),
    },
    {
      value: "guaranteed_throughput",
      label: t("rateLimitTypeFormItem.options.guaranteed"),
      description: t("rateLimitTypeFormItem.options.guaranteedDescription", { lower }),
    },
    {
      value: "dynamic",
      label: t("rateLimitTypeFormItem.options.dynamic"),
      description: t("rateLimitTypeFormItem.options.dynamicDescription", { upper }),
    },
  ];
};

const plainLabels: Record<string, ParseKeys> = {
  best_effort_throughput: "rateLimitTypeFormItem.plainLabels.bestEffort",
  guaranteed_throughput: "rateLimitTypeFormItem.plainLabels.guaranteed",
  dynamic: "rateLimitTypeFormItem.plainLabels.dynamic",
};

const rateLimitTypeLabelText = (type: RateLimitType, t: TFunction): string =>
  t("rateLimitTypeFormItem.label", { type: type.toUpperCase() });

const rateLimitTypeTooltip = (type: RateLimitType, t: TFunction): string =>
  t("rateLimitTypeFormItem.tooltip", { type: type.toUpperCase() });

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
  const { t } = useTranslation();
  const controlId = id ?? `rate-limit-type-${name}`;
  const options = rateLimitTypeOptions(type, t);

  return (
    <div className={className}>
      <TooltipProvider>
        <label htmlFor={controlId} className="mb-2 flex items-center gap-1 text-sm text-foreground">
          {rateLimitTypeLabelText(type, t)}
          <Tooltip>
            <TooltipTrigger
              render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />}
              aria-label={rateLimitTypeTooltip(type, t)}
            />
            <TooltipContent>{rateLimitTypeTooltip(type, t)}</TooltipContent>
          </Tooltip>
        </label>
      </TooltipProvider>
      <Select
        value={value ?? null}
        onValueChange={(next: string | null) => next !== null && onChange?.(next)}
        disabled={disabled}
      >
        <SelectTrigger id={controlId} className="w-full" aria-invalid={ariaInvalid} aria-describedby={ariaDescribedBy}>
          <SelectValue placeholder={t("rateLimitTypeFormItem.placeholder")}>
            {(selected: string | null) =>
              selected !== null && Object.hasOwn(plainLabels, selected)
                ? t(plainLabels[selected])
                : selected ?? t("rateLimitTypeFormItem.placeholder")
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
              <SelectItem key={option.value} value={option.value} title={t(plainLabels[option.value])}>
                {t(plainLabels[option.value])}
              </SelectItem>
            ),
          )}
        </SelectContent>
      </Select>
    </div>
  );
};

export default RateLimitTypeFormItem;
