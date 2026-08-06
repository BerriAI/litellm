import React, { useState } from "react";
import { ExternalLink, Search } from "lucide-react";
import MessageManager from "@/components/molecules/message_manager";
import { searchToolQueryCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

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
    <Card className={`mt-6 ${className}`}>
      <div className="px-6">
        <h2 className="text-lg font-semibold text-foreground">Test Search Tool</h2>
      </div>

      <div className="flex min-h-[600px] flex-col px-6">
        <div className="mb-6">
          <div className="flex items-stretch gap-3">
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute top-1/2 left-3 size-[18px] -translate-y-1/2 text-muted-foreground" />
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSearch();
                  }
                }}
                placeholder="Enter your search query..."
                disabled={isLoading}
                className="h-12 pl-11 text-[15px]"
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading || !query.trim()} className="h-12 px-6 text-[15px]">
              {isLoading ? <UiLoadingSpinner className="size-4" /> : <Search className="size-4" />}
              Search
            </Button>
          </div>
        </div>

        <div className="flex-1">
          {!latestResults && !isLoading ? (
            <div className="flex h-full flex-col items-center justify-center p-8">
              <div className="mb-6 flex size-24 items-center justify-center rounded-full bg-muted">
                <Search className="size-12 text-muted-foreground" />
              </div>
              <p className="text-lg font-medium text-foreground">Test your search tool</p>
              <p className="mt-2 text-sm text-muted-foreground">Enter a query above to see search results</p>
            </div>
          ) : (
            <div>
              {isLoading && (
                <div className="flex flex-col items-center justify-center py-16">
                  <UiLoadingSpinner className="size-8 text-primary" />
                  <p className="mt-4 font-medium text-muted-foreground">Searching...</p>
                </div>
              )}

              {latestResults && !isLoading && (
                <>
                  <div className="mb-6 rounded-lg border border-border bg-muted/50 p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                          Search Query
                        </p>
                        <div className="mt-1.5 text-base font-semibold text-foreground">{latestResults.query}</div>
                      </div>
                      <div className="ml-4 text-right">
                        <p className="text-xs text-muted-foreground">{formatTimestamp(latestResults.timestamp)}</p>
                        <div className="mt-1 flex items-center gap-3">
                          <div className="text-sm font-semibold text-primary">
                            {latestResults.response?.results?.length || 0}{" "}
                            {latestResults.response?.results?.length === 1 ? "result" : "results"}
                          </div>
                          {latestResults.latency !== undefined && (
                            <>
                              <span className="text-muted-foreground">•</span>
                              <div className="text-sm font-semibold text-emerald-600">{latestResults.latency}ms</div>
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {latestResults.response &&
                  latestResults.response.results &&
                  latestResults.response.results.length > 0 ? (
                    <div className="space-y-3">
                      {latestResults.response.results.map((result, resultIndex) => {
                        const isResultExpanded = expandedResults[`0-${resultIndex}`] || false;

                        return (
                          <div
                            key={resultIndex}
                            className="rounded-lg border border-border bg-card transition-shadow hover:shadow-md"
                          >
                            <div className="p-5">
                              <div className="mb-2 flex items-start justify-between gap-3">
                                <a
                                  href={result.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="flex-1 text-lg leading-snug font-semibold text-primary hover:underline"
                                >
                                  {result.title}
                                </a>
                                <Button
                                  variant="ghost"
                                  size="icon-sm"
                                  aria-label="Open result in new tab"
                                  className="shrink-0 text-muted-foreground"
                                  onClick={() => window.open(result.url, "_blank")}
                                >
                                  <ExternalLink className="size-4" />
                                </Button>
                              </div>

                              <div className="mb-3 truncate text-sm font-medium text-emerald-700">{result.url}</div>

                              <div className="text-sm leading-relaxed text-foreground">
                                {isResultExpanded
                                  ? result.snippet
                                  : `${result.snippet.substring(0, 200)}${result.snippet.length > 200 ? "..." : ""}`}
                              </div>

                              {result.snippet.length > 200 && (
                                <Button
                                  variant="link"
                                  size="sm"
                                  className="mt-3 h-auto p-0"
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
                    <div className="rounded-lg border border-border bg-muted/50 py-12 text-center">
                      <div className="mx-auto mb-4 flex size-16 items-center justify-center rounded-full bg-muted">
                        <Search className="size-6 text-muted-foreground" />
                      </div>
                      <p className="font-medium text-foreground">No results found</p>
                      <p className="mt-1 text-sm text-muted-foreground">Try a different search query</p>
                    </div>
                  )}
                </>
              )}

              {searchHistory.length > 1 && (
                <div className="mt-8 border-t border-border pt-6">
                  <div className="mb-4 flex items-center justify-between">
                    <p className="text-sm font-semibold text-foreground">Previous Searches</p>
                    <Button variant="link" size="sm" className="h-auto p-0" onClick={clearHistory}>
                      Clear All
                    </Button>
                  </div>
                  <div className="space-y-2">
                    {searchHistory.slice(1, 6).map((entry, index) => (
                      <div
                        key={index + 1}
                        className="cursor-pointer rounded-lg border border-border bg-muted/50 p-3 transition-colors hover:bg-muted"
                        onClick={() => {
                          setQuery(entry.query);
                        }}
                      >
                        <div className="truncate text-sm font-medium text-foreground">{entry.query}</div>
                        <div className="mt-1.5 flex items-center gap-2 text-xs text-muted-foreground">
                          <span className="font-medium text-primary">
                            {entry.response?.results?.length || 0}{" "}
                            {entry.response?.results?.length === 1 ? "result" : "results"}
                          </span>
                          {entry.latency !== undefined && (
                            <>
                              <span>•</span>
                              <span className="font-medium text-emerald-600">{entry.latency}ms</span>
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
