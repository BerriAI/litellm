import { parseAsString, parseAsStringLiteral, useQueryState } from "nuqs";
import { useCallback } from "react";

const STATUS_FILTERS = ["all", "pending", "active", "rejected"] as const;

const searchParser = parseAsString.withDefault("");
const statusParser = parseAsStringLiteral(STATUS_FILTERS).withDefault("all");
const submissionParser = parseAsString.withOptions({ history: "push" });

export function useTeamGuardrailsUrlState() {
  const [search, setSearchParam] = useQueryState("sub_q", searchParser);
  const [statusFilter, setStatusParam] = useQueryState("sub_status", statusParser);
  const [selectedId, setSelectedParam] = useQueryState("submission", submissionParser);

  const setSearch = useCallback((value: string) => void setSearchParam(value), [setSearchParam]);
  const setStatusFilter = useCallback(
    (value: string) => void setStatusParam(statusParser.parse(value)),
    [setStatusParam],
  );
  const openSubmission = useCallback((id: string) => void setSelectedParam(id), [setSelectedParam]);
  const closeSubmission = useCallback(() => void setSelectedParam(null, { history: "replace" }), [setSelectedParam]);

  return { search, setSearch, statusFilter, setStatusFilter, selectedId, openSubmission, closeSubmission };
}
