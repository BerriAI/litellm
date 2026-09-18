import {
  createParser,
  parseAsArrayOf,
  parseAsInteger,
  parseAsString,
  parseAsStringLiteral,
  useQueryState,
  useQueryStates,
} from "nuqs";

export const ROUTING_GROUP_SORT_COLUMNS = ["group_name", "routing_strategy"] as const;

const boundedInteger = (min: number, max: number) =>
  createParser({
    parse: (value: string) => {
      const parsed = parseAsInteger.parse(value);
      return parsed === null ? null : Math.min(Math.max(parsed, min), max);
    },
    serialize: String,
  });

const pageParser = boundedInteger(1, 100_000).withDefault(1);

const TABLE_URL_STATE = {
  sort_by: parseAsStringLiteral(ROUTING_GROUP_SORT_COLUMNS),
  sort_order: parseAsStringLiteral(["asc", "desc"] as const).withDefault("asc"),
  page: pageParser,
  page_size: boundedInteger(1, 100).withDefault(25),
};

const SEARCH_URL_STATE = {
  group_search: parseAsString.withDefault(""),
  page: pageParser,
};

const expandedGroupsParser = parseAsArrayOf(parseAsString).withDefault([]);

export const useRoutingGroupsTableUrlState = () => useQueryStates(TABLE_URL_STATE);

export const useRoutingGroupsSearchUrlState = () => useQueryStates(SEARCH_URL_STATE);

export const useExpandedRoutingGroups = () => useQueryState("expanded", expandedGroupsParser);
