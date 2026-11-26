import React, { useContext, useId } from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { ChevronDownIcon, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  effectiveClassifierType,
  type ClassifierType,
  type ComplexityRouterConfigValue,
} from "./ComplexityRouterConfig";
import {
  defaultJevClassifierConfig,
  isOssClassifierProvider,
  normalizeJevClassifierConfig,
} from "./jev_classifier_config";
import { transitionClassifierType } from "./classifier_type_transition";
import { isForecastClassifier } from "./forecast_classifier_config";
import {
  AutoRouterAllowanceLabel,
  AutoRouterAvailabilityContext,
  AutoRouterLimits,
  AutoRouterContactLink,
  isAllowanceExhausted,
  AUTO_ROUTER_CONTACT_URL,
} from "./AutoRouterAvailability";
import { productNames } from "@/i18n/untranslated/productNames";

export function ClassifierOption({
  value,
  label,
  description,
  feature,
  disabled,
  unlimited = true,
}: {
  value: string;
  label: string;
  description: string;
  feature?: string;
  disabled?: boolean;
  unlimited?: boolean;
}) {
  const { t } = useTranslation();
  const state = useContext(AutoRouterAvailabilityContext);
  const allowance = state.data?.allowances.find((entry) => entry.key === feature);
  const fresh = !state.isPending && !state.isError && !state.isChecking;
  const exhausted = isAllowanceExhausted(allowance);
  return (
    <div className="relative">
      <DropdownMenuRadioItem value={value} disabled={disabled || exhausted} closeOnClick className="py-3">
        <span className="grid w-full min-w-0 gap-1 whitespace-normal">
          <span className="flex items-center justify-between gap-3">
            <span className="font-medium">{label}</span>
            {feature ? (
              <AutoRouterAllowanceLabel feature={feature} />
            ) : (
              unlimited && (
                <span className="shrink-0 text-xs leading-5 text-muted-foreground">
                  {t("addModel.autoRouterClassifierTabs.option.unlimited")}
                </span>
              )
            )}
          </span>
          <span className={`text-xs leading-5 text-muted-foreground ${exhausted ? "pr-28" : ""}`}>{description}</span>
        </span>
      </DropdownMenuRadioItem>
      {fresh && exhausted && (
        <DropdownMenuItem
          render={<a href={AUTO_ROUTER_CONTACT_URL} target="_blank" rel="noopener noreferrer" />}
          aria-label={t("addModel.autoRouterClassifierTabs.option.talkToTeamAriaLabel", { label })}
          className="absolute top-9 right-8 cursor-pointer px-0 py-0 text-xs leading-5 font-medium text-blue-600 focus:text-blue-600 hover:underline dark:text-blue-400 dark:focus:text-blue-400"
        >
          {t("addModel.autoRouterClassifierTabs.option.talkToTeam")}
        </DropdownMenuItem>
      )}
    </div>
  );
}

export function ClassifierMenu({
  id,
  label,
  value,
  selectedLabel,
  feature,
  onValueChange,
  children,
}: {
  id: string;
  label: string;
  value: string;
  selectedLabel: string;
  feature?: string;
  onValueChange: (value: string) => void;
  children: React.ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        id={id}
        aria-label={label}
        render={<Button variant="outline" className="w-full justify-between font-normal" />}
      >
        <span className="flex-1 text-left">{selectedLabel}</span>
        {feature ? (
          <AutoRouterAllowanceLabel feature={feature} />
        ) : (
          <span className="text-xs text-muted-foreground">{t("addModel.autoRouterClassifierTabs.menu.unlimited")}</span>
        )}
        <ChevronDownIcon className="size-4 text-muted-foreground" />
      </DropdownMenuTrigger>
      <DropdownMenuContent>
        <DropdownMenuRadioGroup value={value} onValueChange={onValueChange}>
          {children}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface AutoRouterClassifierTabsProps {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
  children: React.ReactNode;
  judgeModel?: React.ReactNode;
  decisionModel?: React.ReactNode;
  switchyard?: { selected: boolean; onSelect: () => void; fields: React.ReactNode };
}

const AutoRouterClassifierTabs: React.FC<AutoRouterClassifierTabsProps> = ({
  value,
  onChange,
  children,
  judgeModel,
  decisionModel,
  switchyard,
}) => {
  const { t } = useTranslation();
  const id = useId();
  const availability = useContext(AutoRouterAvailabilityContext);
  const classifierType = effectiveClassifierType(value);
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
  const family = switchyard?.selected ? "switchyard" : familyByType[classifierType];
  const hasCustomTiers = Boolean(value.custom_tier_set);
  const changeType = (next: ClassifierType) => {
    if (next !== classifierType) onChange(transitionClassifierType(value, next));
  };

  const changeFamily = (next: unknown) => {
    if (next === "switchyard") return switchyard?.onSelect();
    if (next === family) return;
    if (next === familyByType[classifierType]) return onChange(value);
    if (next === "heuristics") changeType("heuristic");
    if (next === "llm") changeType("llm");
    if (next === "jev") changeType("jev");
  };
  const changeProvider = (provider: unknown) => {
    if (!isOssClassifierProvider(provider)) return;
    const defaults = defaultJevClassifierConfig(provider);
    onChange({
      ...value,
      jev_classifier_config: { ...defaults, ...value.jev_classifier_config, provider, model: defaults.model },
    });
  };
  const approachLabels: Partial<Record<ClassifierType, string>> = {
    capability: t("addModel.autoRouterClassifierTabs.approach.capability"),
    llm_v2: t("addModel.autoRouterClassifierTabs.approach.fuseV2"),
  };
  const approachDescription: Partial<Record<ClassifierType, string>> = {
    capability: t("addModel.autoRouterClassifierTabs.approach.capabilityHelp"),
    llm_v2: t("addModel.autoRouterClassifierTabs.approach.fuseV2Help"),
  };
  return (
    <div className="flex flex-col gap-4">
      <fieldset>
        <legend className="mb-4 flex w-full items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <h4 className="m-0 text-xl font-semibold text-foreground">
              {t("addModel.autoRouterClassifierTabs.title")}
            </h4>
            <SimpleTooltip content={t("addModel.autoRouterClassifierTabs.titleTooltip")}>
              <button
                type="button"
                aria-label={t("addModel.autoRouterClassifierTabs.aboutClassifiers")}
                className="text-muted-foreground"
              >
                <Info className="size-4" />
              </button>
            </SimpleTooltip>
          </div>
          <AutoRouterLimits />
        </legend>
        <RadioGroup
          value={family}
          onValueChange={changeFamily}
          className={switchyard ? "grid gap-3 sm:grid-cols-2 xl:grid-cols-4" : "grid gap-3 sm:grid-cols-3"}
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
            ...(switchyard
              ? [
                  {
                    value: "switchyard",
                    label: productNames.switchyard,
                    description: t("addModel.autoRouterClassifierTabs.families.switchyardDescription"),
                  },
                ]
              : []),
          ].map((option) => (
            <Label
              key={option.value}
              className="cursor-pointer items-start rounded-lg border p-4 transition-colors hover:bg-muted/50 has-data-checked:border-primary has-data-checked:bg-primary/5"
            >
              <RadioGroupItem
                aria-describedby={`${id}-${option.value}-description`}
                value={option.value}
                disabled={option.value === "heuristics" && hasCustomTiers}
              />
              <span className="space-y-1">
                <span className="block font-medium">{option.label}</span>
                <span
                  id={`${id}-${option.value}-description`}
                  aria-hidden="true"
                  className="block text-xs font-normal text-muted-foreground"
                >
                  {option.description}
                </span>
              </span>
            </Label>
          ))}
        </RadioGroup>
      </fieldset>
      {family === "switchyard" && switchyard?.fields}
      {family === "jev" &&
        (decisionModel ?? (
          <div className="space-y-2">
            <Label htmlFor={`${id}-decision-model`}>{t("addModel.autoRouterClassifierTabs.decisionModelLabel")}</Label>
            <Select
              items={[
                { value: "jev", label: productNames.jev },
                { value: "laya", label: productNames.laya },
                { value: "bespoke", label: productNames.bespokeNimble },
                { value: "databricks", label: productNames.databricks },
                { value: "nadir", label: productNames.nadir },
              ]}
              value={normalizeJevClassifierConfig(value.jev_classifier_config).provider}
              onValueChange={changeProvider}
            >
              <SelectTrigger id={`${id}-decision-model`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="jev">{productNames.jev}</SelectItem>
                <SelectItem value="laya">{productNames.laya}</SelectItem>
                <SelectItem value="bespoke">{productNames.bespokeNimble}</SelectItem>
                <SelectItem value="databricks">{productNames.databricks}</SelectItem>
                <SelectItem value="nadir">{productNames.nadir}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        ))}
      {family === "custom" && (
        <p className="text-sm text-muted-foreground">{t("addModel.autoRouterClassifierTabs.customPlugin")}</p>
      )}
      {family === "heuristics" && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-heuristic`}>{t("addModel.autoRouterClassifierTabs.heuristic.label")}</Label>
          <ClassifierMenu
            id={`${id}-heuristic`}
            label={t("addModel.autoRouterClassifierTabs.heuristic.label")}
            selectedLabel={
              classifierType === "heuristic_v2"
                ? t("addModel.autoRouterClassifierTabs.heuristic.v2")
                : t("addModel.autoRouterClassifierTabs.heuristic.ruleBased")
            }
            feature={classifierType === "heuristic_v2" ? "heuristic_v2" : undefined}
            value={classifierType}
            onValueChange={(next) => {
              if (next === "heuristic" || next === "heuristic_v2") changeType(next);
            }}
          >
            <ClassifierOption
              value="heuristic"
              label={t("addModel.autoRouterClassifierTabs.heuristic.ruleBased")}
              description={t("addModel.autoRouterClassifierTabs.heuristic.ruleBasedDescription")}
            />
            <ClassifierOption
              value="heuristic_v2"
              label={t("addModel.autoRouterClassifierTabs.heuristic.v2")}
              description={t("addModel.autoRouterClassifierTabs.heuristic.v2Description")}
              feature="heuristic_v2"
            />
          </ClassifierMenu>
          <p className="text-xs text-muted-foreground">
            {classifierType === "heuristic_v2"
              ? t("addModel.autoRouterClassifierTabs.heuristic.v2Help")
              : t("addModel.autoRouterClassifierTabs.heuristic.ruleBasedHelp")}
          </p>
        </div>
      )}
      {family === "llm" && judgeModel}
      {family === "llm" && (
        <div className="space-y-2">
          <Label htmlFor={`${id}-approach`}>{t("addModel.autoRouterClassifierTabs.approach.label")}</Label>
          <ClassifierMenu
            id={`${id}-approach`}
            label={t("addModel.autoRouterClassifierTabs.approach.label")}
            selectedLabel={approachLabels[classifierType] ?? t("addModel.autoRouterClassifierTabs.approach.complexity")}
            feature={isForecastClassifier(classifierType) ? classifierType : undefined}
            value={isForecastClassifier(classifierType) ? classifierType : "llm"}
            onValueChange={(next) => {
              if (next === "llm" || next === "capability" || next === "llm_v2") {
                if (next === "llm" && !isForecastClassifier(classifierType)) return;
                changeType(next);
              }
            }}
          >
            <ClassifierOption
              value="llm"
              label={t("addModel.autoRouterClassifierTabs.approach.complexity")}
              description={t("addModel.autoRouterClassifierTabs.approach.complexityDescription")}
            />
            {family === "llm" && (
              <>
                <ClassifierOption
                  value="capability"
                  label={t("addModel.autoRouterClassifierTabs.approach.capability")}
                  description={t("addModel.autoRouterClassifierTabs.approach.capabilityDescription")}
                  feature="capability"
                  disabled={hasCustomTiers}
                />
                <ClassifierOption
                  value="llm_v2"
                  label={t("addModel.autoRouterClassifierTabs.approach.fuseV2")}
                  description={t("addModel.autoRouterClassifierTabs.approach.fuseV2Description")}
                  feature="llm_v2"
                  disabled={hasCustomTiers}
                />
              </>
            )}
          </ClassifierMenu>
          <p className="text-xs text-muted-foreground">
            {approachDescription[classifierType] ?? t("addModel.autoRouterClassifierTabs.approach.complexityHelp")}
          </p>
        </div>
      )}
      {family !== "switchyard" && hasCustomTiers && (
        <p className="text-xs text-muted-foreground">{t("addModel.autoRouterClassifierTabs.restoreStandardTiers")}</p>
      )}
      {availability.data?.error && !availability.isChecking && (
        <div role="alert" className="space-y-1">
          <p className="text-sm text-destructive">{availability.data.error}</p>
          <AutoRouterContactLink />
        </div>
      )}
      {availability.isError && (
        <p className="text-xs text-muted-foreground">
          {t("addModel.autoRouterClassifierTabs.availabilityCheckFailed")}{" "}
          <button type="button" className="underline" onClick={() => availability.refetch?.()}>
            {t("addModel.autoRouterClassifierTabs.retry")}
          </button>
        </p>
      )}
      {family !== "switchyard" && children}
    </div>
  );
};

export default AutoRouterClassifierTabs;
