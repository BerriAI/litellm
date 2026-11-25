import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import {
  choiceToLoggingOnlyScope,
  effectiveLoggingOnlyContinue,
  getLoggingOnlyScopeOptions,
  getLoggingOnlyScopeUpdate,
  type LoggingOnlyScope,
  type LoggingOnlyScopeChoice,
} from "../guardrail_info_helpers";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const LOGGING_ONLY_SCOPE_CHOICES: ReadonlySet<string> = new Set(
  getLoggingOnlyScopeOptions(true).map(({ value }) => value),
);

const isLoggingOnlyScopeChoice = (value: unknown): value is LoggingOnlyScopeChoice =>
  typeof value === "string" && LOGGING_ONLY_SCOPE_CHOICES.has(value);

export const CustomCodeLoggingOnlyScopeSelect: React.FC<{
  value: LoggingOnlyScopeChoice;
  onChange: (choice: LoggingOnlyScopeChoice) => void;
}> = ({ value, onChange }) => {
  const { t } = useTranslation();
  const items = useMemo(() => getLoggingOnlyScopeOptions(true, t), [t]);
  return (
    <div className="w-[240px]">
      <label className="mb-1 block text-xs font-medium text-muted-foreground">
        {t("guardrails.customCode.loggingOnlyScope.label")}
      </label>
      <Select
        items={items}
        value={value}
        onValueChange={(nextValue) => {
          if (isLoggingOnlyScopeChoice(nextValue)) onChange(nextValue);
        }}
      >
        <SelectTrigger className="w-full" aria-label={t("guardrails.customCode.loggingOnlyScope.label")}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {items.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
};

export const getCustomCodeLoggingOnlyScopeCreate = (
  mode: string[],
  choice: LoggingOnlyScopeChoice,
  continueOnInputFailure: boolean,
): { logging_only_scope?: LoggingOnlyScope; logging_only_continue_on_input_failure?: boolean } => {
  if (!mode.includes("logging_only")) return {};
  const scope = choiceToLoggingOnlyScope(choice);
  return {
    ...(scope === null ? {} : { logging_only_scope: scope }),
    ...(effectiveLoggingOnlyContinue(choice, continueOnInputFailure)
      ? { logging_only_continue_on_input_failure: true }
      : {}),
  };
};

export const getCustomCodeLoggingOnlyScopeUpdate = (
  mode: string[],
  litellmParams:
    | { logging_only_scope?: string | null; logging_only_continue_on_input_failure?: boolean | null }
    | null
    | undefined,
  choice: LoggingOnlyScopeChoice,
  continueOnInputFailure: boolean,
): { logging_only_scope?: LoggingOnlyScope | null; logging_only_continue_on_input_failure?: boolean } =>
  mode.includes("logging_only") ? getLoggingOnlyScopeUpdate(litellmParams, choice, continueOnInputFailure) : {};
