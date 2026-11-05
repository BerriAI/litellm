import { useId } from "react";
import { useTranslation } from "react-i18next";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { AutoRouterLimits } from "./AutoRouterAvailability";
import {
  effectiveClassifierType,
  type ClassifierType,
  type ComplexityRouterConfigValue,
} from "./ComplexityRouterConfig";
import { transitionClassifierType } from "./classifier_type_transition";

const familyByType: Record<ClassifierType, string> = {
  heuristic: "heuristics",
  heuristic_v2: "heuristics",
  llm: "llm",
  heuristic_first: "llm",
  hybrid: "llm",
  capability: "llm",
  llm_v2: "llm",
  jev: "jev",
  custom: "custom",
};

export const classifierFamily = (value: ComplexityRouterConfigValue): string =>
  familyByType[effectiveClassifierType(value)];

interface Props {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  switchyardSelected?: boolean;
  onSelectSwitchyard?: () => void;
  allowSwitchyard?: boolean;
}

export default function AutoRouterFamilySelector({
  value,
  onChange,
  switchyardSelected = false,
  onSelectSwitchyard: selectSwitchyard,
  allowSwitchyard = true,
}: Props) {
  const onSelectSwitchyard = allowSwitchyard ? selectSwitchyard : undefined;
  const { t } = useTranslation();
  const id = useId();
  const family = classifierFamily(value);
  const hasCustomTiers = Boolean(value.custom_tier_set);
  const changeFamily = (next: unknown) => {
    if (next === "switchyard") {
      onSelectSwitchyard?.();
      return;
    }
    if (next === family && !switchyardSelected) return;
    if (next === "heuristics") onChange(transitionClassifierType(value, "heuristic"));
    if (next === "llm") onChange(transitionClassifierType(value, "llm"));
    if (next === "jev") onChange(transitionClassifierType(value, "jev"));
  };
  return (
    <fieldset>
      <legend className="mb-4 flex w-full items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <h4 className="m-0 text-xl font-semibold text-foreground">Routing strategy</h4>
          <SimpleTooltip content="Choose how this auto router selects a model for each request.">
            <button type="button" aria-label="About routing strategies" className="text-muted-foreground">
              <Info className="size-4" />
            </button>
          </SimpleTooltip>
        </div>
        <AutoRouterLimits />
      </legend>
      <RadioGroup
        value={switchyardSelected ? "switchyard" : family}
        onValueChange={changeFamily}
        className={onSelectSwitchyard ? "grid gap-3 sm:grid-cols-2 xl:grid-cols-4" : "grid gap-3 sm:grid-cols-3"}
      >
        {[
          {
            value: "heuristics",
            label: t("addModel.autoRouterClassifierTabs.families.heuristics"),
            description: t("addModel.autoRouterClassifierTabs.families.heuristicsDescription"),
          },
          { value: "llm", label: "LLM", description: t("addModel.autoRouterClassifierTabs.families.llmDescription") },
          {
            value: "jev",
            label: t("addModel.autoRouterClassifierTabs.families.decisionModel"),
            description: t("addModel.autoRouterClassifierTabs.families.decisionModelDescription"),
          },
          ...(onSelectSwitchyard
            ? [
                {
                  value: "switchyard",
                  label: "Switchyard",
                  description: "Use tool history to route between capable and efficient models.",
                },
              ]
            : []),
        ].map((option) => (
          <Label
            key={option.value}
            className="min-w-0 cursor-pointer items-start rounded-lg border p-4 transition-colors hover:bg-muted/50 has-data-checked:border-primary has-data-checked:bg-primary/5"
          >
            <RadioGroupItem
              aria-describedby={`${id}-${option.value}-description`}
              value={option.value}
              disabled={option.value === "heuristics" && hasCustomTiers}
            />
            <span className="min-w-0 space-y-1">
              <span className="block whitespace-normal font-medium">{option.label}</span>
              <span
                id={`${id}-${option.value}-description`}
                aria-hidden="true"
                className="block whitespace-normal text-xs font-normal text-muted-foreground"
              >
                {option.description}
              </span>
            </span>
          </Label>
        ))}
      </RadioGroup>
    </fieldset>
  );
}
