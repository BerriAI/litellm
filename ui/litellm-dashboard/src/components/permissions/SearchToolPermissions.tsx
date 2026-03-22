import React, { useState, useEffect } from "react";
import { Text, Badge } from "@tremor/react";
import { SearchIcon } from "@heroicons/react/outline";
import { fetchSearchTools } from "../networking";

interface SearchToolDetails {
  search_tool_id: string;
  search_tool_name?: string;
}

interface SearchToolPermissionsProps {
  searchTools: string[];
  accessToken?: string | null;
}

export function SearchToolPermissions({ searchTools, accessToken }: SearchToolPermissionsProps) {
  const [searchToolDetails, setSearchToolDetails] = useState<SearchToolDetails[]>([]);

  useEffect(() => {
    const loadSearchTools = async () => {
      if (!accessToken || searchTools.length === 0) return;

      try {
        const response = await fetchSearchTools(accessToken);
        if (response.search_tools) {
          setSearchToolDetails(
            response.search_tools.map((tool: any) => ({
              search_tool_id: tool.search_tool_id,
              search_tool_name: tool.search_tool_name,
            })),
          );
        }
      } catch (error) {
        console.error("Error fetching search tools:", error);
      }
    };

    loadSearchTools();
  }, [accessToken, searchTools.length]);

  const getSearchToolDisplayName = (toolId: string) => {
    if (toolId === "*") return "All Search Tools (wildcard)";
    const toolDetail = searchToolDetails.find((tool) => tool.search_tool_id === toolId);
    if (toolDetail) {
      return `${toolDetail.search_tool_name || toolDetail.search_tool_id} (${toolDetail.search_tool_id})`;
    }
    return toolId;
  };

  // Check if this is a wildcard permission (legacy/migrated)
  const isWildcard = searchTools.length === 1 && searchTools[0] === "*";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <SearchIcon className="h-4 w-4 text-amber-600" />
        <Text className="font-semibold text-gray-900">Search Tools</Text>
        <Badge color="amber" size="xs">
          {isWildcard ? "All" : searchTools.length}
        </Badge>
      </div>

      {isWildcard ? (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200">
          <Text className="text-amber-700 text-sm">
            All search tools accessible (migrated permission — consider restricting to specific tools)
          </Text>
        </div>
      ) : searchTools.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {searchTools.map((tool, index) => (
            <div
              key={index}
              className="inline-flex items-center px-3 py-1.5 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-sm font-medium"
            >
              {getSearchToolDisplayName(tool)}
            </div>
          ))}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-gray-50 border border-gray-200">
          <SearchIcon className="h-4 w-4 text-gray-400" />
          <Text className="text-gray-500 text-sm">No search tools configured</Text>
        </div>
      )}
    </div>
  );
}

export default SearchToolPermissions;
