import { Bot, CalendarClock, CircleDashed, SquareChevronRight } from "lucide-react";

import { type ClientIndex, filterItems } from "@/components/shared/search/evaluate";
import { ALL_OPERATORS, type FieldSpec, type QueryLanguage } from "@/components/shared/search/language";
import { scopeLabel } from "../model/format";
import type { Lens } from "../model/types";
import { runStatus } from "../model/status";

const INVESTIGATION_FIELDS = {
  name: { group: "Investigation", icon: SquareChevronRight, suggestValues: true },
  agent: { group: "Investigation", icon: Bot, suggestValues: true },
  status: { group: "Latest run", icon: CircleDashed, suggestValues: true },
  schedule: { group: "Investigation", icon: CalendarClock, suggestValues: true },
} as const satisfies Record<string, FieldSpec>;

export type InvestigationField = keyof typeof INVESTIGATION_FIELDS;

export const INVESTIGATION_QUERY: QueryLanguage<InvestigationField> = {
  fields: INVESTIGATION_FIELDS,
  ops: ALL_OPERATORS,
};

export const INVESTIGATION_INDEX: ClientIndex<Lens, InvestigationField> = {
  read: {
    name: (lens) => [lens.settings.name],
    agent: (lens) => [lens.settings.agent_name, lens.settings.service].filter(Boolean),
    status: (lens) => [lens.jobs[0] ? runStatus(lens.jobs[0]).toLowerCase() : "never"],
    schedule: (lens) => [lens.settings.enabled ? "watching" : "paused"],
  },
  freeText: (lens) => [lens.settings.name, scopeLabel(lens.settings)],
};

export const filterInvestigations = (lenses: Lens[], query: string): Lens[] =>
  filterItems(INVESTIGATION_QUERY, INVESTIGATION_INDEX, lenses, query);
