import React, { useState } from "react";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { Title, Card, Button, Text, Grid } from "@tremor/react";
import { SearchTool, AvailableSearchProvider } from "./types";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { CheckIcon, CopyIcon } from "lucide-react";
import { Button as AntdButton } from "antd";
import { SearchToolTester } from "./search_tool_tester";

interface SearchToolViewProps {
  searchTool: SearchTool;
  onBack: () => void;
  isEditing: boolean;
  accessToken: string | null;
  availableProviders: AvailableSearchProvider[];
}

export const SearchToolView: React.FC<SearchToolViewProps> = ({
  searchTool,
  onBack,
  isEditing,
  accessToken,
  availableProviders,
}) => {
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const getProviderDisplayName = (providerName: string) => {
    const provider = availableProviders.find(p => p.provider_name === providerName);
    return provider?.ui_friendly_name || providerName;
  };

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" className="mb-4" onClick={onBack}>
            Back to All Search Tools
          </Button>
          <div className="flex items-center cursor-pointer">
            <Title>{searchTool.search_tool_name}</Title>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["search-tool-name"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(searchTool.search_tool_name, "search-tool-name")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["search-tool-name"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{searchTool.search_tool_id}</Text>
            <AntdButton
              type="text"
              size="small"
              icon={copiedStates["search-tool-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(searchTool.search_tool_id, "search-tool-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["search-tool-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
      </div>

      <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
        <Card>
          <Text>Provider</Text>
          <div className="mt-2">
            <Title>{getProviderDisplayName(searchTool.litellm_params.search_provider)}</Title>
          </div>
        </Card>

        <Card>
          <Text>API Key</Text>
          <div className="mt-2">
            <Text>{searchTool.litellm_params.api_key ? "****" : "Not set"}</Text>
          </div>
        </Card>

        <Card>
          <Text>Created At</Text>
          <div className="mt-2">
            <Text>
              {searchTool.created_at ? new Date(searchTool.created_at).toLocaleString() : "Unknown"}
            </Text>
          </div>
        </Card>
      </Grid>

      {searchTool.search_tool_info?.description && (
        <Card className="mt-6">
          <Text>Description</Text>
          <div className="mt-2">
            <Text>{searchTool.search_tool_info.description}</Text>
          </div>
        </Card>
      )}

      {/* Search Tool Tester */}
      <div className="mt-6">
        {accessToken && (
          <SearchToolTester
            searchToolName={searchTool.search_tool_name}
            accessToken={accessToken}
          />
        )}
      </div>
    </div>
  );
};


