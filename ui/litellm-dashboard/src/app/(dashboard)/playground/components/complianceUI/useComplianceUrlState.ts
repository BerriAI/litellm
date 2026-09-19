import { parseAsArrayOf, parseAsString, useQueryStates } from "nuqs";
import { useCallback, useState } from "react";

import { useUrlTab } from "@/hooks/useUrlTab";

const RIGHT_PANEL_TABS = ["quick", "batch"] as const;

export type RightPanelTab = (typeof RIGHT_PANEL_TABS)[number];

const NO_SELECTION: string[] = [];

const selectionUrlParsers = {
  cmpl_policies: parseAsArrayOf(parseAsString).withDefault(NO_SELECTION),
  cmpl_guardrails: parseAsArrayOf(parseAsString).withDefault(NO_SELECTION),
};

type ComplianceSelection = {
  cmpl_policies: string[];
  cmpl_guardrails: string[];
};

const EMPTY_SELECTION: ComplianceSelection = {
  cmpl_policies: NO_SELECTION,
  cmpl_guardrails: NO_SELECTION,
};

type SelectionUpdate = (current: ComplianceSelection) => Partial<ComplianceSelection>;

const toggleValue = (values: string[], value: string): string[] =>
  values.includes(value) ? values.filter((current) => current !== value) : [...values, value];

interface ComplianceUrlStateOptions {
  persistInUrl: boolean;
  canViewPolicies: boolean;
}

export function useComplianceUrlState({ persistInUrl, canViewPolicies }: ComplianceUrlStateOptions) {
  const [urlTab, setUrlTab] = useUrlTab(RIGHT_PANEL_TABS, "quick", "cmpl_tab");
  const [urlSelection, setUrlSelection] = useQueryStates(selectionUrlParsers);
  const [localTab, setLocalTab] = useState<RightPanelTab>("quick");
  const [localSelection, setLocalSelection] = useState<ComplianceSelection>(EMPTY_SELECTION);
  const selection = persistInUrl ? urlSelection : localSelection;

  const update = useCallback(
    (change: SelectionUpdate) => {
      if (persistInUrl) {
        void setUrlSelection(change);
        return;
      }
      setLocalSelection((current) => ({ ...current, ...change(current) }));
    },
    [persistInUrl, setUrlSelection],
  );

  const setSelectedPolicies = useCallback(
    (policies: string[]) => update(() => ({ cmpl_policies: policies })),
    [update],
  );

  const toggleGuardrail = useCallback(
    (guardrail: string) => update((current) => ({ cmpl_guardrails: toggleValue(current.cmpl_guardrails, guardrail) })),
    [update],
  );

  const clearSelection = useCallback(() => update(() => EMPTY_SELECTION), [update]);

  return {
    rightTab: persistInUrl ? urlTab : localTab,
    selectedPolicies: canViewPolicies ? selection.cmpl_policies : NO_SELECTION,
    selectedGuardrails: selection.cmpl_guardrails,
    setRightTab: persistInUrl ? setUrlTab : setLocalTab,
    setSelectedPolicies,
    toggleGuardrail,
    clearSelection,
  };
}
