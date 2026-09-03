import { Info } from "lucide-react";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import type { ReasoningEffort } from "./complexity_router_tiers";

const PROVIDER_DEFAULT = "__classifier_provider_default__";

type EffortStatus = "supported" | "unsupported" | "unverified" | undefined;

const effortStatusFor = (
  effort: string | undefined,
  explicitlySupported: string[] | null | undefined,
): EffortStatus => {
  if (effort === undefined) return undefined;
  if (!Array.isArray(explicitlySupported)) return "unverified";
  return explicitlySupported.includes(effort) ? "supported" : "unsupported";
};

export const classifierEffortOptionsForModels = (
  modelInfo: ModelGroup[],
): Record<string, string[] | null | undefined> =>
  Object.fromEntries(modelInfo.map((model) => [model.model_group, model.supported_reasoning_efforts]));

interface ClassifierReasoningEffortSelectProps {
  model: string;
  value: ReasoningEffort | undefined;
  explicitlySupported: string[] | null | undefined;
  onChange: (value: ReasoningEffort | undefined) => void;
}

const ClassifierReasoningEffortSelect = ({
  model,
  value,
  explicitlySupported,
  onChange,
}: ClassifierReasoningEffortSelectProps) => {
  const status = effortStatusFor(value, explicitlySupported);
  const options = Array.from(new Set([...(explicitlySupported ?? []), ...(value ? [value] : [])]));

  if (!model || options.length === 0) return null;

  const optionLabel = (effort: string): string =>
    effort === value && status !== "supported" ? `${effort} (${status})` : effort;

  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <strong className="font-semibold">Reasoning Effort</strong>
        <SimpleTooltip content="Sent only to the classifier call. Default leaves the classifier deployment or provider setting unchanged.">
          <Info className="size-4 text-muted-foreground" />
        </SimpleTooltip>
      </div>
      <Select
        items={[
          { value: PROVIDER_DEFAULT, label: "Default" },
          ...options.map((effort) => ({ value: effort, label: optionLabel(effort) })),
        ]}
        value={value ?? PROVIDER_DEFAULT}
        onValueChange={(effort: string | null) =>
          effort && onChange(effort === PROVIDER_DEFAULT ? undefined : (effort as ReasoningEffort))
        }
      >
        <SelectTrigger aria-label={`Reasoning effort for classifier model ${model}`} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={PROVIDER_DEFAULT}>Default</SelectItem>
          {options.map((effort) => (
            <SelectItem key={effort} value={effort}>
              {optionLabel(effort)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {status === "unverified" && (
        <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">
          This saved effort cannot be verified for the selected model. Choose Default unless you have confirmed provider
          support.
        </p>
      )}
      {status === "unsupported" && (
        <p className="mt-1 text-xs text-destructive">
          This saved effort is not supported by every deployment in the selected model group. Choose Default or a
          supported value before saving.
        </p>
      )}
    </div>
  );
};

export default ClassifierReasoningEffortSelect;
