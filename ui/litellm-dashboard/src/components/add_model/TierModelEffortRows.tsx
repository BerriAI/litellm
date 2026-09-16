import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Switch } from "@/components/ui/switch";
import { Info } from "lucide-react";
import React from "react";
import { ReasoningEffort, TierModelParams } from "./complexity_router_tiers";

const PROVIDER_DEFAULT = "__provider_default__";

const storedEffort = (params: TierModelParams | undefined): ReasoningEffort | undefined => {
  const stored = params?.reasoning_effort;
  if (stored === undefined || stored === null || stored === "") return undefined;
  return typeof stored === "string" ? stored : String(stored);
};

interface TierModelEffortRowsProps {
  tierLabel: string;
  models: string[];
  effortOptionsByModel: Record<string, string[]>;
  paramsByModel: Record<string, TierModelParams> | undefined;
  onEffortChange: (model: string, effort: ReasoningEffort | undefined) => void;
  fastModeByModel?: Record<string, boolean>;
  onFastModeChange: (model: string, enabled: boolean) => void;
}

export interface TierEffortRow {
  model: string;
  effort: ReasoningEffort | undefined;
  options: string[];
}

/**
 * A stored effort outside the model's supported set (hand-authored, or capabilities changed since
 * it was saved) is listed anyway, so the row renders with its value selected and can be cleared.
 */
export const tierEffortRows = ({
  models,
  effortOptionsByModel,
  paramsByModel,
  fastModeByModel,
}: Pick<
  TierModelEffortRowsProps,
  "models" | "effortOptionsByModel" | "paramsByModel" | "fastModeByModel"
>): TierEffortRow[] =>
  models
    .map((model) => {
      const effort = storedEffort(paramsByModel?.[model]);
      const supported = effortOptionsByModel[model] ?? [];
      const listed = effort !== undefined && !supported.includes(effort) ? [...supported, effort] : supported;
      return { model, effort, options: Array.from(new Set(listed)) };
    })
    .filter(({ model, options }) => options.length > 0 || fastModeByModel?.[model] === true);

const TierModelEffortRows: React.FC<TierModelEffortRowsProps> = (props) => {
  const { tierLabel, paramsByModel, onEffortChange, fastModeByModel, onFastModeChange } = props;
  const rows = tierEffortRows(props);
  if (rows.length === 0) return null;
  return (
    <div className="mt-2 space-y-1">
      {rows.some(({ options }) => options.length > 0) && (
        <div className="flex items-center gap-1">
          <span className="text-xs font-medium text-muted-foreground">Reasoning effort</span>
          <SimpleTooltip
            content={`Sent as reasoning_effort on requests this tier routes to the model, overriding the caller's value. Default leaves the request untouched.`}
          >
            <Info className="size-3 text-muted-foreground/70" />
          </SimpleTooltip>
        </div>
      )}
      {rows.map(({ model, effort, options }) => (
        <div key={model} className="flex flex-wrap items-center justify-between gap-2">
          <span className="min-w-0 flex-1 basis-32 truncate text-xs" title={model}>
            {model}
          </span>
          <div className="flex flex-wrap items-center gap-3">
            {options.length > 0 && (
              <Select
                items={[
                  { value: PROVIDER_DEFAULT, label: "Default" },
                  ...options.map((option) => ({ value: option, label: option })),
                ]}
                value={effort ?? PROVIDER_DEFAULT}
                onValueChange={(selected: string | null) =>
                  selected !== null && onEffortChange(model, selected === PROVIDER_DEFAULT ? undefined : selected)
                }
              >
                <SelectTrigger
                  size="sm"
                  className="w-36"
                  aria-label={`Reasoning effort for ${model} in the ${tierLabel} tier`}
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={PROVIDER_DEFAULT}>Default</SelectItem>
                  {options.map((option) => (
                    <SelectItem key={option} value={option}>
                      {option}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            {fastModeByModel?.[model] === true && (
              <SimpleTooltip content="Fast mode has higher pricing and requires an eligible provider account. Off removes this tier's speed override and inherits the request or provider default">
                <label
                  className="flex items-center gap-2 text-xs"
                  aria-label={`Fast mode for ${model} in the ${tierLabel} tier`}
                >
                  <Switch
                    size="sm"
                    checked={paramsByModel?.[model]?.speed === "fast"}
                    onCheckedChange={(enabled) => onFastModeChange(model, enabled)}
                  />
                  Fast mode
                </label>
              </SimpleTooltip>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default TierModelEffortRows;
