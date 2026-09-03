import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
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
}

export interface TierEffortRow {
  model: string;
  effort: ReasoningEffort | undefined;
  options: string[];
}

/**
 * A stored effort outside the model's supported set (hand-authored, or capabilities changed since
 * it was saved) is listed anyway, so the row renders with its value selected and can be cleared.
 * Only a model with no supported level and nothing stored drops out.
 */
export const tierEffortRows = ({
  models,
  effortOptionsByModel,
  paramsByModel,
}: Pick<TierModelEffortRowsProps, "models" | "effortOptionsByModel" | "paramsByModel">): TierEffortRow[] =>
  models
    .map((model) => {
      const effort = storedEffort(paramsByModel?.[model]);
      const supported = effortOptionsByModel[model] ?? [];
      const listed = effort !== undefined && !supported.includes(effort) ? [...supported, effort] : supported;
      return { model, effort, options: Array.from(new Set(listed)) };
    })
    .filter(({ options }) => options.length > 0);

const TierModelEffortRows: React.FC<TierModelEffortRowsProps> = ({
  tierLabel,
  models,
  effortOptionsByModel,
  paramsByModel,
  onEffortChange,
}) => {
  const rows = tierEffortRows({ models, effortOptionsByModel, paramsByModel });
  if (rows.length === 0) return null;
  return (
    <div className="mt-2 space-y-1">
      <div className="flex items-center gap-1">
        <span className="text-xs font-medium text-muted-foreground">Reasoning effort</span>
        <SimpleTooltip
          content={`Sent as reasoning_effort on requests this tier routes to the model, overriding the caller's value. Default leaves the request untouched.`}
        >
          <Info className="size-3 text-muted-foreground/70" />
        </SimpleTooltip>
      </div>
      {rows.map(({ model, effort, options }) => (
        <div key={model} className="flex items-center justify-between gap-2">
          <span className="truncate text-xs">{model}</span>
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
        </div>
      ))}
    </div>
  );
};

export default TierModelEffortRows;
