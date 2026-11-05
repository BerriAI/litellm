import { CircleHelp, Minus, Plus } from "lucide-react";
import React from "react";
import type { ParseKeys } from "i18next";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

import NumericalInput from "../shared/numerical_input";

export const CACHE_CONTROL_LABEL: ParseKeys = "addModel.cacheControlSettings.label";

export const CACHE_CONTROL_TOOLTIP: ParseKeys = "addModel.cacheControlSettings.tooltip";

export const CACHE_CONTROL_DESCRIPTION: ParseKeys = "addModel.cacheControlSettings.description";

export const CACHE_CONTROL_ROLE_HINT: ParseKeys = "addModel.cacheControlSettings.roleHint";

export const CACHE_CONTROL_INDEX_HINT: ParseKeys = "addModel.cacheControlSettings.indexHint";

export type CacheControlRole = "user" | "system" | "assistant";

export interface CacheControlInjectionPoint {
  location: "message";
  role?: CacheControlRole;
  index?: string | number;
}

export const NEW_CACHE_CONTROL_POINT: CacheControlInjectionPoint = { location: "message" };

const LOCATION_ITEMS = [{ value: "message", labelKey: "addModel.cacheControlSettings.locations.message" }] as const;

const ROLE_ITEMS = [
  { value: "user", labelKey: "addModel.cacheControlSettings.roles.user" },
  { value: "system", labelKey: "addModel.cacheControlSettings.roles.system" },
  { value: "assistant", labelKey: "addModel.cacheControlSettings.roles.assistant" },
] as const;

const LabelWithHint: React.FC<{ label: string; hint: string }> = ({ label, hint }) => {
  const { t } = useTranslation();
  return (
    <div className="flex items-center">
      <Label>{label}</Label>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <button
                type="button"
                aria-label={t("addModel.cacheControlSettings.labelHelp", { label })}
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
  const { t } = useTranslation();
  const points = value ?? [];

  const replaceAt = (index: number, point: CacheControlInjectionPoint) =>
    onChange?.(points.map((existing, position) => (position === index ? point : existing)));

  return (
    <div className="ml-6 border-l-2 border-border pl-4">
      <p className="mb-4 block text-sm text-muted-foreground">{t(CACHE_CONTROL_DESCRIPTION)}</p>

      {points.map((point, index) => (
        <div key={index} className="mb-4 flex items-end gap-4">
          <div className="w-[180px] space-y-1">
            <Label>{t("addModel.cacheControlSettings.type")}</Label>
            <Select
              items={LOCATION_ITEMS.map((item) => ({ value: item.value, label: t(item.labelKey) }))}
              value={point.location}
              disabled
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {LOCATION_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {t(item.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="w-[180px] space-y-1">
            <LabelWithHint label={t("addModel.cacheControlSettings.role")} hint={t(CACHE_CONTROL_ROLE_HINT)} />
            <Select
              items={ROLE_ITEMS.map((item) => ({ value: item.value, label: t(item.labelKey) }))}
              value={point.role ?? null}
              onValueChange={(selected) =>
                replaceAt(index, { ...point, role: (selected as CacheControlRole | null) ?? undefined })
              }
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder={t("addModel.cacheControlSettings.selectRole")} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={null}>{t("addModel.cacheControlSettings.noRole")}</SelectItem>
                {ROLE_ITEMS.map((item) => (
                  <SelectItem key={item.value} value={item.value}>
                    {t(item.labelKey)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="w-[180px] space-y-1">
            <LabelWithHint label={t("addModel.cacheControlSettings.index")} hint={t(CACHE_CONTROL_INDEX_HINT)} />
            <NumericalInput
              type="number"
              placeholder={t("addModel.cacheControlSettings.optional")}
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
              aria-label={t("addModel.cacheControlSettings.removePoint", { position: index + 1 })}
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
        {t("addModel.cacheControlSettings.addPoint")}
      </Button>
    </div>
  );
};

export default CacheControlInjectionPoints;
