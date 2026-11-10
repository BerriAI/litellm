import type { SkillListQuery } from "./types";
import type { SortingState } from "@tanstack/react-table";
import { parseAsString, parseAsStringLiteral, useQueryStates } from "nuqs";
import { useUrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable/useUrlTableState";
import { matchesSearchTerm } from "@/utils/searchUtils";

const SCAN_FILTERS = ["all", "yes", "no"] as const;
export type ScanFilter = (typeof SCAN_FILTERS)[number];

export interface SkillCatalogMetadata {
  created_by?: string | null;
  approved_by?: string | null;
  published_by?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  security_scan_passed?: boolean;
  download_count?: number | null;
  keywords?: string[] | null;
}

export const SKILL_SORT_FIELDS = [
  "name",
  "download_count",
  "created_at",
  "updated_at",
  "created_by",
  "approved_by",
  "security_scan_passed",
] as const satisfies readonly NonNullable<SkillListQuery["sort_by"]>[];

export function skillSortQuery(sorting: SortingState): Pick<SkillListQuery, "sort_by" | "sort_order"> {
  return {
    sort_by: SKILL_SORT_FIELDS.find((field) => field === sorting[0]?.id) ?? "name",
    sort_order: sorting[0]?.desc ? "desc" : "asc",
  };
}

const CATALOG_STATE: UrlTableStateOptions<never> = {
  keyPrefix: "skills_",
  defaultPageSize: 50,
  maxPageSize: 100,
  sortFields: SKILL_SORT_FIELDS,
  defaultSort: { id: "name", desc: false },
  filterColumns: [],
};
const FILTER_PARSERS = {
  tag: parseAsString.withDefault(""),
  scan: parseAsStringLiteral(SCAN_FILTERS).withDefault("all"),
};
const FILTER_OPTIONS = { urlKeys: { tag: "skills_filter_tag", scan: "skills_filter_scan" } };

export function useSkillFilters(sortFields: readonly string[] = SKILL_SORT_FIELDS) {
  const table = useUrlTableState({ ...CATALOG_STATE, sortFields });
  const [filters, setFilters] = useQueryStates(FILTER_PARSERS, FILTER_OPTIONS);
  const update = (next: { tag?: string; scan?: ScanFilter } | null) => {
    void setFilters(next);
    table.onPaginationChange({ ...table.pagination, pageIndex: 0 });
  };
  return {
    ...table,
    ...filters,
    setTag: (tag: string) => update({ tag }),
    setScan: (scan: ScanFilter) => update({ scan }),
    clear: () => {
      update(null);
      table.setSearch("");
    },
    active: Boolean(table.search.trim() || filters.tag || filters.scan !== "all"),
  };
}

export function skillTags(skills: SkillCatalogMetadata[]): string[] {
  return [...new Set(skills.flatMap((skill) => skill.keywords ?? []))].sort((a, b) => a.localeCompare(b));
}

export function matchesSkillFilters(
  skill: SkillCatalogMetadata & {
    name: string;
    description?: string | null;
    domain?: string | null;
    namespace?: string | null;
  },
  filters: { search: string; tag: string; scan: ScanFilter },
): boolean {
  const matchesQuery = matchesSearchTerm(filters.search, [
    skill.name,
    skill.description,
    skill.domain,
    skill.namespace,
    skill.created_by,
    skill.approved_by,
    skill.published_by,
    ...(skill.keywords ?? []),
  ]);
  const matchesTag =
    !filters.tag || (skill.keywords ?? []).some((tag) => tag.toLowerCase() === filters.tag.toLowerCase());
  const matchesScan = filters.scan === "all" || Boolean(skill.security_scan_passed) === (filters.scan === "yes");
  return matchesQuery && matchesTag && matchesScan;
}
