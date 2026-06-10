import type { TFunction } from "i18next";
import FilterTeamDropdown from "../common_components/FilterTeamDropdown";
import { PaginatedKeyAliasSelect } from "../KeyAliasSelect/PaginatedKeyAliasSelect/PaginatedKeyAliasSelect";
import { PaginatedModelSelect } from "../ModelSelect/PaginatedModelSelect/PaginatedModelSelect";
import { FilterOption } from "../molecules/filter";
import { allEndUsersCall } from "../networking";
import { getErrorCodeOptions } from "./constants";
import { FILTER_KEYS } from "./log_filter_logic";

export function getLogFilterOptions(accessToken: string, t: TFunction): FilterOption[] {
  return [
    {
      name: FILTER_KEYS.TEAM_ID,
      label: t("viewLogs.filterOptions.teamIdLabel"),
      customComponent: FilterTeamDropdown,
    },
    {
      name: FILTER_KEYS.STATUS,
      label: t("viewLogs.filterOptions.statusLabel"),
      isSearchable: false,
      options: [
        { label: t("viewLogs.filterOptions.statusSuccess"), value: "success" },
        { label: t("viewLogs.filterOptions.statusFailure"), value: "failure" },
      ],
    },
    {
      name: FILTER_KEYS.KEY_ALIAS,
      label: t("viewLogs.filterOptions.keyAliasLabel"),
      customComponent: PaginatedKeyAliasSelect,
    },
    {
      name: FILTER_KEYS.END_USER,
      label: t("viewLogs.filterOptions.endUserLabel"),
      isSearchable: true,
      searchFn: async (searchText: string) => {
        const data = await allEndUsersCall(accessToken);
        const users = data?.map((u: any) => u.user_id) || [];
        const filtered = users.filter((u: string) => u.toLowerCase().includes(searchText.toLowerCase()));
        return filtered.map((u: string) => ({ label: u, value: u }));
      },
    },
    {
      name: FILTER_KEYS.ERROR_CODE,
      label: t("viewLogs.filterOptions.errorCodeLabel"),
      isSearchable: true,
      searchFn: async (searchText: string) => {
        const errorCodeOptions = getErrorCodeOptions(t);
        if (!searchText) return errorCodeOptions;
        const lower = searchText.toLowerCase();
        const filtered = errorCodeOptions.filter((opt) => opt.label.toLowerCase().includes(lower));
        const isExactValue = errorCodeOptions.some((opt) => opt.value === searchText.trim());
        if (!isExactValue && searchText.trim()) {
          filtered.push({
            label: t("viewLogs.filterOptions.customCodeLabel", { code: searchText.trim() }),
            value: searchText.trim(),
          });
        }
        return filtered;
      },
    },
    {
      name: FILTER_KEYS.ERROR_MESSAGE,
      label: t("viewLogs.filterOptions.errorMessageLabel"),
      isSearchable: false,
    },
    {
      name: FILTER_KEYS.KEY_HASH,
      label: t("viewLogs.filterOptions.keyHashLabel"),
      isSearchable: false,
    },
    {
      name: FILTER_KEYS.MODEL,
      label: t("viewLogs.filterOptions.modelLabel"),
      customComponent: PaginatedModelSelect,
    },
    {
      name: FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL,
      label: t("viewLogs.filterOptions.publicModelLabel"),
      isSearchable: false,
    },
  ];
}
