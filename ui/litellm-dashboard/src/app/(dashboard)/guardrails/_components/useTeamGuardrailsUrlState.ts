import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useCallback } from "react";

const STATUS_FILTERS = ["all", "pending", "active", "rejected"] as const;
const SUBMITTED_TAB = "submitted";

const statusParser = parseAsStringLiteral(STATUS_FILTERS).withDefault("all");

const parsers = {
  sub_q: parseAsString.withDefault(""),
  sub_status: statusParser,
  submission: parseAsString.withOptions({ history: "push" }),
  tab: parseAsStringLiteral([SUBMITTED_TAB] as const),
};

export function useTeamGuardrailsUrlState() {
  const [{ sub_q: search, sub_status: statusFilter, submission: selectedId }, setParams] = useQueryStates(parsers);

  const setSearch = useCallback((value: string) => void setParams({ sub_q: value, tab: SUBMITTED_TAB }), [setParams]);
  const setStatusFilter = useCallback(
    (value: string) => void setParams({ sub_status: statusParser.parse(value), tab: SUBMITTED_TAB }),
    [setParams],
  );
  const openSubmission = useCallback(
    (id: string) => void setParams({ submission: id, tab: SUBMITTED_TAB }),
    [setParams],
  );
  const closeSubmission = useCallback(
    () => void setParams({ submission: null, tab: SUBMITTED_TAB }, { history: "replace" }),
    [setParams],
  );

  return { search, setSearch, statusFilter, setStatusFilter, selectedId, openSubmission, closeSubmission };
}
