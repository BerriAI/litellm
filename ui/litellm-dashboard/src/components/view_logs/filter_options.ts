import FilterTeamDropdown from "../common_components/FilterTeamDropdown";
import { PaginatedKeyAliasSelect } from "../KeyAliasSelect/PaginatedKeyAliasSelect/PaginatedKeyAliasSelect";
import { PaginatedModelSelect } from "../ModelSelect/PaginatedModelSelect/PaginatedModelSelect";
import { FilterOption } from "../molecules/filter";
import { allEndUsersCall } from "../networking";
import { ERROR_CODE_OPTIONS } from "./constants";
import { FILTER_KEYS } from "./log_filter_logic";

export function getLogFilterOptions(accessToken: string): FilterOption[] {
  return [
    {
      name: "Team ID",
      label: "Team ID",
      customComponent: FilterTeamDropdown,
    },
    {
      name: "Status",
      label: "Status",
      isSearchable: false,
      options: [
        { label: "Success", value: "success" },
        { label: "Failure", value: "failure" },
      ],
    },
    {
      name: "Key Alias",
      label: "Key Alias",
      customComponent: PaginatedKeyAliasSelect,
    },
    {
      name: "End User",
      label: "End User",
      isSearchable: true,
      searchFn: async (searchText: string) => {
        const data = await allEndUsersCall(accessToken);
        const users = data?.map((u: any) => u.user_id) || [];
        const filtered = users.filter((u: string) => u.toLowerCase().includes(searchText.toLowerCase()));
        return filtered.map((u: string) => ({ label: u, value: u }));
      },
    },
    {
      name: "Error Code",
      label: "Error Code",
      isSearchable: true,
      searchFn: async (searchText: string) => {
        if (!searchText) return ERROR_CODE_OPTIONS;
        const lower = searchText.toLowerCase();
        const filtered = ERROR_CODE_OPTIONS.filter((opt) => opt.label.toLowerCase().includes(lower));
        const isExactValue = ERROR_CODE_OPTIONS.some((opt) => opt.value === searchText.trim());
        if (!isExactValue && searchText.trim()) {
          filtered.push({ label: `Use custom code: ${searchText.trim()}`, value: searchText.trim() });
        }
        return filtered;
      },
    },
    {
      name: "Error Message",
      label: "Error Message",
      isSearchable: false,
    },
    {
      name: "Key Hash",
      label: "Key Hash",
      isSearchable: false,
    },
    {
      name: "Model",
      label: "Model",
      customComponent: PaginatedModelSelect,
    },
    {
      name: FILTER_KEYS.PUBLIC_MODEL_OR_SEARCH_TOOL,
      label: "Public model / search tool",
      isSearchable: false,
    },
  ];
}
