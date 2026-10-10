import React from "react";
import {
  choiceToLoggingOnlyScope,
  effectiveLoggingOnlyContinue,
  getLoggingOnlyScopeOptions,
  getLoggingOnlyScopeUpdate,
  type LoggingOnlyScope,
  type LoggingOnlyScopeChoice,
} from "../guardrail_info_helpers";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const LOGGING_ONLY_SCOPE_ITEMS = getLoggingOnlyScopeOptions(true);
const LOGGING_ONLY_SCOPE_CHOICES: ReadonlySet<string> = new Set(LOGGING_ONLY_SCOPE_ITEMS.map(({ value }) => value));

const isLoggingOnlyScopeChoice = (value: unknown): value is LoggingOnlyScopeChoice =>
  typeof value === "string" && LOGGING_ONLY_SCOPE_CHOICES.has(value);

export const CustomCodeLoggingOnlyScopeSelect: React.FC<{
  value: LoggingOnlyScopeChoice;
  onChange: (choice: LoggingOnlyScopeChoice) => void;
}> = ({ value, onChange }) => (
  <div className="w-[240px]">
    <label className="mb-1 block text-xs font-medium text-muted-foreground">Logging only scope</label>
    <Select
      items={LOGGING_ONLY_SCOPE_ITEMS}
      value={value}
      onValueChange={(nextValue) => {
        if (isLoggingOnlyScopeChoice(nextValue)) onChange(nextValue);
      }}
    >
      <SelectTrigger className="w-full" aria-label="Logging only scope">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {LOGGING_ONLY_SCOPE_ITEMS.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  </div>
);

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
