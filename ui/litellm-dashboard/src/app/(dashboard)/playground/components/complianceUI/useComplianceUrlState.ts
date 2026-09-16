import { parseAsArrayOf, parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback, useState } from "react";

const RIGHT_PANEL_TABS = ["quick", "batch"] as const;

export type RightPanelTab = (typeof RIGHT_PANEL_TABS)[number];

const NO_SELECTION: string[] = [];

const complianceUrlParsers = {
  cmpl_tab: parseAsStringLiteral(RIGHT_PANEL_TABS).withDefault("quick"),
  cmpl_policies: parseAsArrayOf(parseAsString).withDefault(NO_SELECTION),
  cmpl_guardrails: parseAsArrayOf(parseAsString).withDefault(NO_SELECTION),
};

type ComplianceUrlValues = {
  cmpl_tab: RightPanelTab;
  cmpl_policies: string[];
  cmpl_guardrails: string[];
};

const DEFAULT_VALUES: ComplianceUrlValues = {
  cmpl_tab: "quick",
  cmpl_policies: NO_SELECTION,
  cmpl_guardrails: NO_SELECTION,
};

type ComplianceUrlUpdate = (current: ComplianceUrlValues) => Partial<ComplianceUrlValues>;

const toggleValue = (values: string[], value: string): string[] =>
  values.includes(value) ? values.filter((current) => current !== value) : [...values, value];

export function useComplianceUrlState(persistInUrl: boolean) {
  const [urlValues, setUrlValues] = useQueryStates(complianceUrlParsers);
  const [localValues, setLocalValues] = useState<ComplianceUrlValues>(DEFAULT_VALUES);
  const values = persistInUrl ? urlValues : localValues;

  const update = useCallback(
    (change: ComplianceUrlUpdate) => {
      if (persistInUrl) {
        void setUrlValues(change);
        return;
      }
      setLocalValues((current) => ({ ...current, ...change(current) }));
    },
    [persistInUrl, setUrlValues],
  );

  const setRightTab = useCallback((tab: RightPanelTab) => update(() => ({ cmpl_tab: tab })), [update]);

  const setSelectedPolicies = useCallback(
    (policies: string[]) => update(() => ({ cmpl_policies: policies })),
    [update],
  );

  const toggleGuardrail = useCallback(
    (guardrail: string) => update((current) => ({ cmpl_guardrails: toggleValue(current.cmpl_guardrails, guardrail) })),
    [update],
  );

  const clearSelection = useCallback(
    () => update(() => ({ cmpl_policies: NO_SELECTION, cmpl_guardrails: NO_SELECTION })),
    [update],
  );

  return {
    rightTab: values.cmpl_tab,
    selectedPolicies: values.cmpl_policies,
    selectedGuardrails: values.cmpl_guardrails,
    setRightTab,
    setSelectedPolicies,
    toggleGuardrail,
    clearSelection,
  };
}
