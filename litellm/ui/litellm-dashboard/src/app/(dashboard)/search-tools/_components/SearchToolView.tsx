import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";
import { ArrowLeft, Check, Copy } from "lucide-react";
import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
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
  isEditing,
  accessToken,
  availableProviders,
}) => {
  const { t } = useTranslation();
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
    const provider = availableProviders.find((p) => p.provider_name === providerName);
    return provider?.ui_friendly_name || providerName;
  };

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" size="sm" className="mb-4 -ml-2 text-muted-foreground" onClick={onBack}>
            <ArrowLeft className="mr-2 size-4" />
            {t("searchTools.view.back")}
          </Button>
          <div className="flex items-center gap-1">
            <h1 className="text-2xl font-semibold text-foreground">{searchTool.search_tool_name}</h1>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={t("searchTools.view.copyName")}
              className="text-muted-foreground"
              onClick={() => copyToClipboard(searchTool.search_tool_name, "search-tool-name")}
            >
              {copiedStates["search-tool-name"] ? <Check /> : <Copy />}
            </Button>
          </div>
          <div className="flex items-center gap-1">
            <p className="font-mono text-sm text-muted-foreground">{searchTool.search_tool_id}</p>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label={t("searchTools.view.copyId")}
              className="text-muted-foreground"
              onClick={() => copyToClipboard(searchTool.search_tool_id, "search-tool-id")}
            >
              {copiedStates["search-tool-id"] ? <Check /> : <Copy />}
            </Button>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">{t("searchTools.view.provider")}</p>
            <p className="mt-2 text-lg font-semibold text-foreground">
              {getProviderDisplayName(searchTool.litellm_params.search_provider)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">{t("searchTools.view.apiKey")}</p>
            <p className="mt-2 text-foreground">
              {searchTool.litellm_params.api_key ? "****" : t("searchTools.view.apiKeyNotSet")}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">{t("searchTools.view.cost")}</p>
            <p className="mt-2 text-foreground">
              {typeof searchTool.litellm_params.input_cost_per_query === "number"
                ? `$${searchTool.litellm_params.input_cost_per_query}`
                : t("searchTools.view.costDefault")}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">{t("searchTools.view.createdAt")}</p>
            <p className="mt-2 text-foreground">
              {searchTool.created_at
                ? new Date(searchTool.created_at).toLocaleString()
                : t("searchTools.view.createdUnknown")}
            </p>
          </CardContent>
        </Card>
      </div>

      {searchTool.search_tool_info?.description && (
        <Card className="mt-6">
          <CardContent>
            <p className="text-sm text-muted-foreground">{t("searchTools.view.description")}</p>
            <p className="mt-2 text-foreground">{searchTool.search_tool_info.description}</p>
          </CardContent>
        </Card>
      )}

      <div className="mt-6">
        {accessToken && <SearchToolTester searchToolName={searchTool.search_tool_name} accessToken={accessToken} />}
      </div>
    </div>
  );
};
