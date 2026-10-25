import { useLanguage } from "@/contexts/LanguageContext";
import { CircleHelp, Minus, Plus } from "lucide-react";
import React from "react";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import NumericalInput from "../shared/numerical_input";

export const CACHE_CONTROL_LABEL = "Cache Control Injection Points";

export const CACHE_CONTROL_TOOLTIP =
  "Tell litellm where to inject cache control checkpoints. You can specify either by role (to apply to all messages of that role) or by specific message index.";

export const CACHE_CONTROL_DESCRIPTION =
  "Providers like Anthropic, Bedrock API require users to specify where to inject cache control checkpoints, litellm can automatically add them for you as a cost saving feature.";

export const CACHE_CONTROL_ROLE_HINT = "LiteLLM will mark all messages of this role as cacheable";

export const CACHE_CONTROL_INDEX_HINT = "(Optional) If set litellm will mark the message at this index as cacheable";

export type CacheControlRole = "user" | "system" | "assistant";

export interface CacheControlInjectionPoint {
  location: "message";
  role?: CacheControlRole;
  index?: string | number;
}

export const NEW_CACHE_CONTROL_POINT: CacheControlInjectionPoint = { location: "message" };

const LOCATION_ITEMS = [{ value: "message", label: "Message" }] as const;

const ROLE_ITEMS = [
  { value: "user", label: "User" },
  { value: "system", label: "System" },
  { value: "assistant", label: "Assistant" },
] as const;

const LabelWithHint: React.FC<{ label: string; hint: string }> = ({ label, hint }) => {
  const { t } = useLanguage();
  return (
    <div className="flex items-center">
      <Label>{label}</Label>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <button
                type="button"
                aria-label={t("{label} help", { label })}
                className="ml-1 inline-flex cursor-help items-center rounded-sm text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              />
            }
          >
            <CircleHelp aria-hidden className="size-4" />
          </TooltipTrigger>
          <TooltipContent className="max-w-xs whitespace-normal">{hint}</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

interface CacheControlInjectionPointsProps {
  value?: CacheControlInjectionPoint[];
  onChange?: (points: CacheControlInjectionPoint[]) => void;
}

/**
 * Editor for `cache_control_injection_points`. It holds no form state of its own so that an antd
 * `Form.Item` and a react-hook-form `FormField` can each host it while their pages migrate
 * independently; both hand a child exactly `value` and `onChange`.
 */
const CacheControlInjectionPoints: React.FC<CacheControlInjectionPointsProps> = ({ value, onChange }) => {
  const { t } = useLanguage();
  const points = value ?? [];

  const replaceAt = (index: number, point: CacheControlInjectionPoint) =>
    onChange?.(points.map((existing, position) => (position === index ? point : existing)));

  return (
    <div className="ml-6 border-l-2 border-border pl-4">
      <p className="mb-4 block text-sm text-muted-foreground">{t(CACHE_CONTROL_DESCRIPTION)}</p>

      {points.map((point, index) => (
        <div key={index} className="mb-4 flex items-end gap-4">
          <div className="w-[180px] space-y-1">
            <Label>{t("Type")}</Label>
            <Select
              items={LOCATION_ITEMS.map((item) => ({ ...item, label: t(item.label) }))}
              value={point.location}
              disabled
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LOCATION_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {t(item.label)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="w-[180px] space-y-1">
            <LabelWithHint label={t("Role")} hint={t(CACHE_CONTROL_ROLE_HINT)} />
            <Select
              items={ROLE_ITEMS.map((item) => ({ ...item, label: t(item.label) }))}
              value={point.role ?? null}
              onValueChange={(selected) =>
                replaceAt(index, { ...point, role: (selected as CacheControlRole | null) ?? undefined })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder={t("Select a role")} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={null}>{t("None")}</SelectItem>
                {ROLE_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {t(item.label)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="w-[180px] space-y-1">
            <LabelWithHint label={t("Index")} hint={t(CACHE_CONTROL_INDEX_HINT)} />
            <NumericalInput
              type="number"
              placeholder={t("Optional")}
              step={1}
              value={point.index ?? ""}
              onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                replaceAt(index, {
                  ...point,
                  index: event.target.value === "" ? undefined : event.target.value,
                })
              }
            />
          </div>

          {points.length > 1 && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={t("Remove injection point {index}", { index: index + 1 })}
              className="text-destructive"
              onClick={() => onChange?.(points.filter((_, position) => position !== index))}
            >
              <Minus className="size-4" />
            </Button>
          )}
        </div>
      ))}

      <Button
        type="button"
        variant="outline"
        className="w-full border-dashed"
        onClick={() => onChange?.([...points, NEW_CACHE_CONTROL_POINT])}
      >
        <Plus className="mr-2 size-4" />
        {t("Add Injection Point")}
      </Button>
    </div>
  );
};

export default CacheControlInjectionPoints;
