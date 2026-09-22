import { useUrlTableState, type UrlTableState, type UrlTableStateOptions } from "@/components/shared/DataTable";

const FILTER_COLUMNS = ["org_id"] as const;
type FilterColumn = (typeof FILTER_COLUMNS)[number];

const TABLE_STATE_OPTIONS: UrlTableStateOptions<FilterColumn> = {
  sortFields: ["organization_id", "organization_alias", "created_at", "spend"],
  defaultSort: { id: "created_at", desc: true },
  defaultPageSize: 25,
  filterColumns: FILTER_COLUMNS,
  urlKeys: { search: "org_search" },
};

export const useOrganizationsTableState = (): UrlTableState => useUrlTableState(TABLE_STATE_OPTIONS);

export const organizationIdFilter = ({ columnFilters }: Pick<UrlTableState, "columnFilters">): string => {
  const value = columnFilters.find((filter) => filter.id === "org_id")?.value;
  return typeof value === "string" ? value : "";
};
