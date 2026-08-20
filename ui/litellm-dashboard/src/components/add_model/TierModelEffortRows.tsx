import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import React from "react";
import { REASONING_EFFORT_OPTIONS, ReasoningEffort, TierModelParams } from "./complexity_router_tiers";

const PROVIDER_DEFAULT = "__provider_default__";

const asEffort = (params: TierModelParams | undefined): ReasoningEffort | undefined => {
  const stored = params?.reasoning_effort;
  if (typeof stored !== "string") return undefined;
  return REASONING_EFFORT_OPTIONS.find((option) => option === stored);
};

interface TierModelEffortRowsProps {
  tierLabel: string;
  models: string[];
  reasoningModels: ReadonlySet<string>;
  paramsByModel: Record<string, TierModelParams> | undefined;
  onEffortChange: (model: string, effort: ReasoningEffort | undefined) => void;
}

const TierModelEffortRows: React.FC<TierModelEffortRowsProps> = ({
  tierLabel,
  models,
  reasoningModels,
  paramsByModel,
  onEffortChange,
}) => {
  const shown = models.filter(
    (model) => reasoningModels.has(model) || Object.keys(paramsByModel?.[model] ?? {}).length > 0,
  );
  if (shown.length === 0) return null;
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
      {shown.map((model) => (
        <div key={model} className="flex items-center justify-between gap-2">
          <span className="truncate text-xs">{model}</span>
          <Select
            items={[
              { value: PROVIDER_DEFAULT, label: "Default" },
              ...REASONING_EFFORT_OPTIONS.map((option) => ({ value: option, label: option })),
            ]}
            value={asEffort(paramsByModel?.[model]) ?? PROVIDER_DEFAULT}
            onValueChange={(selected: string | null) =>
              selected !== null &&
              onEffortChange(model, selected === PROVIDER_DEFAULT ? undefined : (selected as ReasoningEffort))
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
              {REASONING_EFFORT_OPTIONS.map((option) => (
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
