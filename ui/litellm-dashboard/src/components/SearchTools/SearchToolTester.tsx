import React, { useState } from "react";
import { Button, Input, Typography, Spin } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { SearchOutlined, LoadingOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { searchToolQueryCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { Card, Title as TremorTitle } from "@tremor/react";

const { Text } = Typography;

interface SearchResult {
  title: string;
  url: string;
  snippet: string;
}

interface SearchToolQueryResponse {
  results: SearchResult[];
}

interface SearchToolTesterProps {
  searchToolName: string;
  accessToken: string;
  className?: string;
}

export const SearchToolTester: React.FC<SearchToolTesterProps> = ({ searchToolName, accessToken, className = "" }) => {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [searchHistory, setSearchHistory] = useState<
    {
      query: string;
      response: SearchToolQueryResponse | null;
      timestamp: number;
      latency?: number;
    }[]
  >([]);
  const [expandedResults, setExpandedResults] = useState<Record<string, boolean>>({});
  const [isInputFocused, setIsInputFocused] = useState(false);

  const handleSearch = async () => {
    if (!query.trim()) {
      MessageManager.warning(t("searchTools.searchToolTester.enterQueryWarning"));
      return;
    }

    setIsLoading(true);
    const startTime = performance.now();

    try {
      const response = await searchToolQueryCall(accessToken, searchToolName, query);
      const endTime = performance.now();
      const latency = Math.round(endTime - startTime);

      const historyEntry = {
        query,
        response,
        timestamp: Date.now(),
        latency,
      };

      setSearchHistory((prev) => [historyEntry, ...prev]);
      // Don't clear query after search so user can modify it
    } catch (error) {
      console.error("Error querying search tool:", error);
      NotificationsManager.fromBackend(t("searchTools.searchToolTester.queryFailed"));
    } finally {
      setIsLoading(false);
    }
  };

  const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp).toLocaleString();
  };

  const clearHistory = () => {
    setSearchHistory([]);
    setExpandedResults({});
    NotificationsManager.success(t("searchTools.searchToolTester.historyCleared"));
  };

  const toggleResultExpansion = (historyIndex: number, resultIndex: number) => {
    const key = `${historyIndex}-${resultIndex}`;
    setExpandedResults((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const antIcon = <LoadingOutlined style={{ fontSize: 24 }} spin />;

  const latestResults = searchHistory.length > 0 ? searchHistory[0] : null;

  return (
    <Card className="mt-6">
      <div className="mb-6">
        <TremorTitle>{t("searchTools.searchToolTester.title")}</TremorTitle>
      </div>

      <div className="flex flex-col" style={{ minHeight: "600px" }}>
        {/* Search Bar at Top */}
        <div className="mb-6">
          <div className="flex items-stretch gap-3">
            <div
              className="flex items-center flex-1 bg-white rounded-lg px-4 transition-all duration-200"
              style={{
                border: isInputFocused ? "2px solid #3b82f6" : "2px solid #e5e7eb",
                boxShadow: isInputFocused ? "0 0 0 3px rgba(59, 130, 246, 0.1)" : "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                height: "48px",
              }}
            >
              <SearchOutlined className="text-gray-400 mr-3" style={{ fontSize: "18px" }} />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => setIsInputFocused(true)}
                onBlur={() => setIsInputFocused(false)}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault();
                    handleSearch();
                  }
                }}
                placeholder={t("searchTools.searchToolTester.searchPlaceholder")}
                disabled={isLoading}
                bordered={false}
                style={{ fontSize: "15px", padding: 0, height: "100%", boxShadow: "none" }}
              />
            </div>
            <Button
              type="primary"
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              icon={<SearchOutlined />}
              loading={isLoading}
              style={{
                height: "48px",
                paddingLeft: "24px",
                paddingRight: "24px",
                borderRadius: "8px",
                fontWeight: 500,
                fontSize: "15px",
                backgroundColor: isLoading || !query.trim() ? undefined : "#1890ff",
                borderColor: isLoading || !query.trim() ? undefined : "#1890ff",
                boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
              }}
            >
              {t("common.search")}
            </Button>
          </div>
        </div>

        {/* Results Area */}
        <div className="flex-1">
          {!latestResults && !isLoading ? (
            <div className="h-full flex flex-col items-center justify-center p-8">
              <div className="flex items-center justify-center w-24 h-24 rounded-full bg-gray-100 mb-6">
                <SearchOutlined style={{ fontSize: "48px", color: "#9ca3af" }} />
              </div>
              <Text className="text-lg text-gray-600 font-medium">
                {t("searchTools.searchToolTester.emptyStateTitle")}
              </Text>
              <Text className="text-sm text-gray-500 mt-2">{t("searchTools.searchToolTester.emptyStateSubtitle")}</Text>
            </div>
          ) : (
            <div>
              {isLoading && (
                <div className="flex flex-col justify-center items-center py-16">
                  <Spin indicator={antIcon} />
                  <Text className="mt-4 text-gray-600 font-medium">{t("searchTools.searchToolTester.searching")}</Text>
                </div>
              )}

              {latestResults && !isLoading && (
                <>
                  {/* Query Info Bar */}
                  <div
                    className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg"
                    style={{ boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)" }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <Text className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                          {t("searchTools.searchToolTester.searchQueryLabel")}
                        </Text>
                        <div className="text-base font-semibold text-gray-900 mt-1.5">{latestResults.query}</div>
                      </div>
                      <div className="text-right ml-4">
                        <Text className="text-xs text-gray-500">{formatTimestamp(latestResults.timestamp)}</Text>
                        <div className="flex items-center gap-3 mt-1">
                          <div className="text-sm font-semibold text-blue-600">
                            {t("searchTools.searchToolTester.resultCount", {
                              count: latestResults.response?.results?.length || 0,
                            })}
                          </div>
                          {latestResults.latency !== undefined && (
                            <>
                              <span className="text-gray-400">•</span>
                              <div className="text-sm font-semibold text-green-600">{latestResults.latency}ms</div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Search Results */}
                  {latestResults.response &&
                  latestResults.response.results &&
                  latestResults.response.results.length > 0 ? (
                    <div className="space-y-3">
                      {latestResults.response.results.map((result, resultIndex) => {
                        const isResultExpanded = expandedResults[`0-${resultIndex}`] || false;

                        return (
                          <div
                            key={resultIndex}
                            className="bg-white border border-gray-200 rounded-lg overflow-hidden transition-all duration-200"
                            style={{ boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)" }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.boxShadow =
                                "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)";
                              e.currentTarget.style.borderColor = "#e0e7ff";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.boxShadow = "0 1px 2px 0 rgba(0, 0, 0, 0.05)";
                              e.currentTarget.style.borderColor = "#e5e7eb";
                            }}
                          >
                            <div className="p-5">
                              {/* Title and External Link */}
                              <div className="flex items-start justify-between gap-3 mb-2">
                                <a
                                  href={result.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-lg font-semibold text-blue-600 hover:text-blue-700 flex-1 leading-snug"
                                  style={{ textDecoration: "none" }}
                                  onMouseEnter={(e) => (e.currentTarget.style.textDecoration = "underline")}
                                  onMouseLeave={(e) => (e.currentTarget.style.textDecoration = "none")}
                                >
                                  {result.title}
                                </a>
                                <Button
                                  type="text"
                                  size="small"
                                  className="flex-shrink-0"
                                  icon={
                                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                                      />
                                    </svg>
                                  }
                                  onClick={() => window.open(result.url, "_blank")}
                                  style={{ color: "#6b7280" }}
                                />
                              </div>

                              {/* URL */}
                              <div className="text-sm text-green-700 mb-3 truncate font-medium">{result.url}</div>

                              {/* Snippet Preview */}
                              <div className="text-sm text-gray-700 leading-relaxed">
                                {isResultExpanded
                                  ? result.snippet
                                  : `${result.snippet.substring(0, 200)}${result.snippet.length > 200 ? "..." : ""}`}
                              </div>

                              {/* Expand/Collapse */}
                              {result.snippet.length > 200 && (
                                <Button
                                  type="link"
                                  size="small"
                                  className="mt-3 p-0 h-auto"
                                  onClick={() => toggleResultExpansion(0, resultIndex)}
                                  style={{
                                    fontSize: "13px",
                                    fontWeight: 500,
                                    color: "#3b82f6",
                                  }}
                                >
                                  {isResultExpanded
                                    ? t("searchTools.searchToolTester.showLess")
                                    : t("searchTools.searchToolTester.showMore")}
                                </Button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center py-12 bg-gray-50 border border-gray-200 rounded-lg">
                      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-gray-100 mx-auto mb-4">
                        <SearchOutlined style={{ fontSize: "24px", color: "#9ca3af" }} />
                      </div>
                      <Text className="text-gray-600 font-medium">{t("common.noResults")}</Text>
                      <Text className="text-sm text-gray-500 mt-1">
                        {t("searchTools.searchToolTester.tryDifferentQuery")}
                      </Text>
                    </div>
                  )}
                </>
              )}

              {/* Search History Sidebar */}
              {searchHistory.length > 1 && (
                <div className="mt-8 pt-6 border-t border-gray-200">
                  <div className="flex items-center justify-between mb-4">
                    <Text className="text-sm font-semibold text-gray-700">
                      {t("searchTools.searchToolTester.previousSearches")}
                    </Text>
                    <Button
                      onClick={clearHistory}
                      size="small"
                      type="link"
                      style={{
                        fontSize: "13px",
                        fontWeight: 500,
                      }}
                    >
                      {t("searchTools.searchToolTester.clearAll")}
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {searchHistory.slice(1, 6).map((entry, index) => (
                      <div
                        key={index + 1}
                        className="p-3 bg-gray-50 border border-gray-200 rounded-lg cursor-pointer transition-all duration-200 hover:bg-gray-100 hover:border-gray-300"
                        onClick={() => {
                          setQuery(entry.query);
                        }}
                      >
                        <div className="text-sm font-medium text-gray-800 truncate">{entry.query}</div>
                        <div className="text-xs text-gray-500 mt-1.5 flex items-center gap-2">
                          <span className="font-medium text-blue-600">
                            {t("searchTools.searchToolTester.resultCount", {
                              count: entry.response?.results?.length || 0,
                            })}
                          </span>
                          {entry.latency !== undefined && (
                            <>
                              <span>•</span>
                              <span className="font-medium text-green-600">{entry.latency}ms</span>
                            </>
                          )}
                          <span>•</span>
                          <span>{formatTimestamp(entry.timestamp)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  );
};

export default SearchToolTester;
