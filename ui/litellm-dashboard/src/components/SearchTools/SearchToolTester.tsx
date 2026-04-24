import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import MessageManager from "@/components/molecules/message_manager";
import { Loader2, Search as SearchOutlined } from "lucide-react";
import { searchToolQueryCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

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
      MessageManager.warning("Please enter a search query");
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
    } catch (error) {
      console.error("Error querying search tool:", error);
      NotificationsManager.fromBackend("Failed to query search tool");
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
    NotificationsManager.success("Search history cleared");
  };

  const toggleResultExpansion = (historyIndex: number, resultIndex: number) => {
    const key = `${historyIndex}-${resultIndex}`;
    setExpandedResults((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const latestResults = searchHistory.length > 0 ? searchHistory[0] : null;

  return (
    <Card className={`mt-6 p-6 ${className}`}>
      <div className="mb-6">
        <h3 className="text-lg font-semibold">Test Search Tool</h3>
      </div>

      <div className="flex flex-col" style={{ minHeight: "600px" }}>
        {/* Search Bar at Top */}
        <div className="mb-6">
          <div className="flex items-stretch gap-3">
            <div
              className="flex items-center flex-1 bg-background rounded-lg px-4 transition-all duration-200"
              style={{
                border: isInputFocused ? "2px solid #3b82f6" : "2px solid #e5e7eb",
                boxShadow: isInputFocused ? "0 0 0 3px rgba(59, 130, 246, 0.1)" : "0 1px 2px 0 rgba(0, 0, 0, 0.05)",
                height: "48px",
              }}
            >
              <SearchOutlined className="text-muted-foreground mr-3 h-[18px] w-[18px]" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onFocus={() => setIsInputFocused(true)}
                onBlur={() => setIsInputFocused(false)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSearch();
                  }
                }}
                placeholder="Enter your search query..."
                disabled={isLoading}
                className="border-0 shadow-none focus-visible:ring-0 focus-visible:ring-offset-0 text-[15px] h-full p-0"
              />
            </div>
            <Button
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              style={{
                height: "48px",
                paddingLeft: "24px",
                paddingRight: "24px",
                fontSize: "15px",
              }}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <SearchOutlined className="h-4 w-4" />
              )}
              Search
            </Button>
          </div>
        </div>

        {/* Results Area */}
        <div className="flex-1">
          {!latestResults && !isLoading ? (
            <div className="h-full flex flex-col items-center justify-center p-8">
              <div className="flex items-center justify-center w-24 h-24 rounded-full bg-muted mb-6">
                <SearchOutlined className="h-12 w-12 text-muted-foreground" />
              </div>
              <p className="text-lg text-foreground font-medium">Test your search tool</p>
              <p className="text-sm text-muted-foreground mt-2">Enter a query above to see search results</p>
            </div>
          ) : (
            <div>
              {isLoading && (
                <div className="flex flex-col justify-center items-center py-16">
                  <Loader2 className="h-6 w-6 animate-spin" />
                  <p className="mt-4 text-muted-foreground font-medium">Searching...</p>
                </div>
              )}

              {latestResults && !isLoading && (
                <>
                  {/* Query Info Bar */}
                  <div
                    className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg dark:bg-blue-950/30 dark:border-blue-900"
                    style={{ boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)" }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                          Search Query
                        </p>
                        <div className="text-base font-semibold text-foreground mt-1.5">{latestResults.query}</div>
                      </div>
                      <div className="text-right ml-4">
                        <p className="text-xs text-muted-foreground">{formatTimestamp(latestResults.timestamp)}</p>
                        <div className="flex items-center gap-3 mt-1">
                          <div className="text-sm font-semibold text-blue-600">
                            {latestResults.response?.results?.length || 0}{" "}
                            {latestResults.response?.results?.length === 1 ? "result" : "results"}
                          </div>
                          {latestResults.latency !== undefined && (
                            <>
                              <span className="text-muted-foreground">•</span>
                              <div className="text-sm font-semibold text-green-600">{latestResults.latency}ms</div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Search Results */}
                  {latestResults.response && latestResults.response.results && latestResults.response.results.length > 0 ? (
                    <div className="space-y-3">
                      {latestResults.response.results.map((result, resultIndex) => {
                        const isResultExpanded = expandedResults[`0-${resultIndex}`] || false;

                        return (
                          <div
                            key={resultIndex}
                            className="bg-background border border-border rounded-lg overflow-hidden transition-all duration-200"
                            style={{ boxShadow: "0 1px 2px 0 rgba(0, 0, 0, 0.05)" }}
                            onMouseEnter={(e) => {
                              e.currentTarget.style.boxShadow =
                                "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)";
                            }}
                            onMouseLeave={(e) => {
                              e.currentTarget.style.boxShadow = "0 1px 2px 0 rgba(0, 0, 0, 0.05)";
                            }}
                          >
                            <div className="p-5">
                              {/* Title and External Link */}
                              <div className="flex items-start justify-between gap-3 mb-2">
                                <a
                                  href={result.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-lg font-semibold text-blue-600 hover:text-blue-700 hover:underline flex-1 leading-snug"
                                >
                                  {result.title}
                                </a>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="flex-shrink-0 text-muted-foreground"
                                  onClick={() => window.open(result.url, "_blank")}
                                  aria-label="Open in new tab"
                                >
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                                    />
                                  </svg>
                                </Button>
                              </div>

                              {/* URL */}
                              <div className="text-sm text-green-700 mb-3 truncate font-medium">{result.url}</div>

                              {/* Snippet Preview */}
                              <div className="text-sm text-foreground leading-relaxed">
                                {isResultExpanded
                                  ? result.snippet
                                  : `${result.snippet.substring(0, 200)}${result.snippet.length > 200 ? "..." : ""}`}
                              </div>

                              {/* Expand/Collapse */}
                              {result.snippet.length > 200 && (
                                <Button
                                  variant="link"
                                  size="sm"
                                  className="mt-3 p-0 h-auto text-blue-600"
                                  onClick={() => toggleResultExpansion(0, resultIndex)}
                                >
                                  {isResultExpanded ? "Show less" : "Show more"}
                                </Button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="text-center py-12 bg-muted border border-border rounded-lg">
                      <div className="flex items-center justify-center w-16 h-16 rounded-full bg-muted mx-auto mb-4">
                        <SearchOutlined className="h-6 w-6 text-muted-foreground" />
                      </div>
                      <p className="text-muted-foreground font-medium">No results found</p>
                      <p className="text-sm text-muted-foreground mt-1">Try a different search query</p>
                    </div>
                  )}
                </>
              )}

              {/* Search History */}
              {searchHistory.length > 1 && (
                <div className="mt-8 pt-6 border-t border-border">
                  <div className="flex items-center justify-between mb-4">
                    <p className="text-sm font-semibold text-foreground">Previous Searches</p>
                    <Button onClick={clearHistory} variant="link" size="sm" className="text-sm">
                      Clear All
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {searchHistory.slice(1, 6).map((entry, index) => (
                      <div
                        key={index + 1}
                        className="p-3 bg-muted border border-border rounded-lg cursor-pointer transition-all duration-200 hover:bg-muted/80"
                        onClick={() => {
                          setQuery(entry.query);
                        }}
                      >
                        <div className="text-sm font-medium text-foreground truncate">{entry.query}</div>
                        <div className="text-xs text-muted-foreground mt-1.5 flex items-center gap-2">
                          <span className="font-medium text-blue-600">
                            {entry.response?.results?.length || 0}{" "}
                            {entry.response?.results?.length === 1 ? "result" : "results"}
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
