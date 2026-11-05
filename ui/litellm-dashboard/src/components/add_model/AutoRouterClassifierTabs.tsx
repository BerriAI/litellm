import AutoRouterFamilySelector, { classifierFamily } from "./AutoRouterFamilySelector";
import React, { useContext, useId } from "react";
import { useTranslation } from "react-i18next";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { ChevronDownIcon } from "lucide-react";
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
  AutoRouterContactLink,
  isAllowanceExhausted,
  AUTO_ROUTER_CONTACT_URL,
} from "./AutoRouterAvailability";

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
  showFamilySelector?: boolean;
}

const AutoRouterClassifierTabs: React.FC<AutoRouterClassifierTabsProps> = ({
  value,
  onChange,
  children,
  judgeModel,
  decisionModel,
  showFamilySelector = true,
}) => {
  const { t } = useTranslation();
  const id = useId();
  const availability = useContext(AutoRouterAvailabilityContext);
  const classifierType = effectiveClassifierType(value);
  const family = classifierFamily(value);
  const hasCustomTiers = Boolean(value.custom_tier_set);
  const changeType = (next: ClassifierType) => {
    if (next !== classifierType) onChange(transitionClassifierType(value, next));
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
      {showFamilySelector && <AutoRouterFamilySelector value={value} onChange={onChange} />}
      {family === "jev" &&
        (decisionModel ?? (
          <div className="space-y-2">
            <Label htmlFor={`${id}-decision-model`}>{t("addModel.autoRouterClassifierTabs.decisionModelLabel")}</Label>
            <Select
              items={[
                { value: "jev", label: "Jev" },
                { value: "laya", label: "Laya" },
                { value: "bespoke", label: "Bespoke Nimble" },
                { value: "databricks", label: "Databricks" },
              ]}
              value={normalizeJevClassifierConfig(value.jev_classifier_config).provider}
              onValueChange={changeProvider}
            >
              <SelectTrigger id={`${id}-decision-model`} className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="jev">Jev</SelectItem>
                <SelectItem value="laya">Laya</SelectItem>
                <SelectItem value="bespoke">Bespoke Nimble</SelectItem>
                <SelectItem value="databricks">Databricks</SelectItem>
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
      {hasCustomTiers && (
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
      {children}
    </div>
  );
};

export default AutoRouterClassifierTabs;
