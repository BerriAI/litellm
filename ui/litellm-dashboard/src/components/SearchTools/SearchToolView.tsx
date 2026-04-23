import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ArrowLeft, CheckIcon, CopyIcon } from "lucide-react";
import React, { useState } from "react";
import { SearchToolTester } from "./SearchToolTester";
import { AvailableSearchProvider, SearchTool } from "./types";

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
  accessToken,
  availableProviders,
}) => {
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});

  const copyToClipboard = async (
    text: string | null | undefined,
    key: string,
  ) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const getProviderDisplayName = (providerName: string) => {
    const provider = availableProviders.find(
      (p) => p.provider_name === providerName,
    );
    return provider?.ui_friendly_name || providerName;
  };

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" size="sm" className="mb-4" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
            Back to All Search Tools
          </Button>
          <div className="flex items-center gap-1">
            <h2 className="text-2xl font-semibold">
              {searchTool.search_tool_name}
            </h2>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() =>
                copyToClipboard(
                  searchTool.search_tool_name,
                  "search-tool-name",
                )
              }
              aria-label="Copy search tool name"
            >
              {copiedStates["search-tool-name"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-muted-foreground font-mono">
              {searchTool.search_tool_id}
            </span>
            <Button
              variant="ghost"
              size="icon"
              className="h-7 w-7"
              onClick={() =>
                copyToClipboard(searchTool.search_tool_id, "search-tool-id")
              }
              aria-label="Copy search tool id"
            >
              {copiedStates["search-tool-id"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card className="p-4">
          <p className="text-sm">Provider</p>
          <div className="mt-2">
            <h3 className="text-lg font-semibold">
              {getProviderDisplayName(
                searchTool.litellm_params.search_provider,
              )}
            </h3>
          </div>
        </Card>

        <Card className="p-4">
          <p className="text-sm">API Key</p>
          <div className="mt-2">
            <p className="text-sm">
              {searchTool.litellm_params.api_key ? "****" : "Not set"}
            </p>
          </div>
        </Card>

        <Card className="p-4">
          <p className="text-sm">Created At</p>
          <div className="mt-2">
            <p className="text-sm">
              {searchTool.created_at
                ? new Date(searchTool.created_at).toLocaleString()
                : "Unknown"}
            </p>
          </div>
        </Card>
      </div>

      {searchTool.search_tool_info?.description && (
        <Card className="mt-6 p-4">
          <p className="text-sm">Description</p>
          <div className="mt-2">
            <p className="text-sm">{searchTool.search_tool_info.description}</p>
          </div>
        </Card>
      )}

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
