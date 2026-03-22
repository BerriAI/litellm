import React, { useState } from "react";
import { Select, Typography } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { SearchToolTester } from "./SearchToolTester";
import { SearchTool, AvailableSearchProvider } from "./types";

const { Text } = Typography;

interface SearchToolTestPlaygroundProps {
  searchTools: SearchTool[];
  availableProviders: AvailableSearchProvider[];
  isLoading: boolean;
  accessToken: string;
}

const SearchToolTestPlayground: React.FC<SearchToolTestPlaygroundProps> = ({
  searchTools,
  availableProviders,
  isLoading,
  accessToken,
}) => {
  const [selectedToolName, setSelectedToolName] = useState<string | null>(null);

  const getProviderDisplayName = (providerName: string) => {
    const provider = availableProviders.find((p) => p.provider_name === providerName);
    return provider?.ui_friendly_name || providerName;
  };

  return (
    <div className="w-full">
      <div className="mb-6">
        <Text className="text-sm text-gray-600 mb-3 block">
          Select a search tool to test with live queries.
        </Text>
        <Select
          placeholder="Select a search tool to test"
          className="w-full"
          size="large"
          value={selectedToolName}
          onChange={setSelectedToolName}
          loading={isLoading}
          showSearch
          optionFilterProp="label"
          allowClear
          options={searchTools.map((tool) => ({
            label: `${tool.search_tool_name} (${getProviderDisplayName(tool.litellm_params.search_provider)})`,
            value: tool.search_tool_name,
          }))}
        />
      </div>

      {selectedToolName ? (
        <SearchToolTester
          searchToolName={selectedToolName}
          accessToken={accessToken}
        />
      ) : (
        <div className="flex flex-col items-center justify-center py-16 text-gray-400">
          <SearchOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
          <Text className="text-lg font-medium text-gray-600 mb-2">
            Select a Search Tool to Test
          </Text>
          <Text className="text-center text-gray-500 max-w-md">
            Choose a search tool from the dropdown above to start testing queries and viewing results.
          </Text>
        </div>
      )}
    </div>
  );
};

export default SearchToolTestPlayground;
