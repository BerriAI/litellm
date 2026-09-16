import { parseAsArrayOf, parseAsString, useQueryState } from "nuqs";

const expandedGroupsParser = parseAsArrayOf(parseAsString).withDefault([]);

export const useExpandedRoutingGroups = () => useQueryState("expanded", expandedGroupsParser);
