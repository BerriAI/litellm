import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import MessageManager from "@/components/molecules/message_manager";
import {
  ChevronDown,
  ChevronRight,
  Database,
  Loader2,
  Send,
} from "lucide-react";
import { vectorStoreSearchCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

interface VectorStoreContent {
  text: string;
  type: string;
}

interface VectorStoreResult {
  score: number;
  content: VectorStoreContent[];
  file_id?: string;
  filename?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  attributes?: Record<string, any>;
}

interface VectorStoreSearchResponse {
  object: string;
  search_query: string;
  data: VectorStoreResult[];
}

interface VectorStoreTesterProps {
  vectorStoreId: string;
  accessToken: string;
  className?: string;
}

export const VectorStoreTester: React.FC<VectorStoreTesterProps> = ({
  vectorStoreId,
  accessToken,
}) => {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [searchHistory, setSearchHistory] = useState<
    {
      query: string;
      response: VectorStoreSearchResponse | null;
      timestamp: number;
    }[]
  >([]);
  const [expandedResults, setExpandedResults] = useState<
    Record<string, boolean>
  >({});

  const handleSearch = async () => {
    if (!query.trim()) {
      MessageManager.warning("Please enter a search query");
      return;
    }

    setIsLoading(true);

    try {
      const response = await vectorStoreSearchCall(
        accessToken,
        vectorStoreId,
        query,
      );

      const historyEntry = {
        query,
        response,
        timestamp: Date.now(),
      };

      setSearchHistory((prev) => [historyEntry, ...prev]);
      setQuery("");
    } catch (error) {
      console.error("Error searching vector store:", error);
      NotificationsManager.fromBackend("Failed to search vector store");
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSearch();
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

  return (
    <Card className="w-full rounded-xl shadow-md">
      <div className="flex flex-col h-[600px]">
        {/* Header */}
        <div className="p-4 border-b border-border flex justify-between items-center">
          <div className="flex items-center">
            {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
            <Database className="mr-2 text-blue-500 h-4 w-4" />
            <h4 className="text-base font-semibold m-0">
              Test Vector Store
            </h4>
          </div>
          {searchHistory.length > 0 && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={clearHistory}
            >
              Clear History
            </Button>
          )}
        </div>

        {/* Results Area */}
        <div className="flex-1 overflow-auto p-4 pb-0">
          {searchHistory.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
              <Database className="h-12 w-12 mb-4" />
              <span>
                Test your vector store by entering a search query below
              </span>
            </div>
          ) : (
            <div className="space-y-4">
              {searchHistory.map((entry, index) => (
                <div key={index} className="space-y-2">
                  {/* User Query */}
                  <div className="text-right">
                    {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                    <div className="inline-block max-w-[80%] rounded-lg shadow-sm p-3 bg-blue-50 border border-blue-200 dark:bg-blue-950/30 dark:border-blue-900">
                      <div className="flex items-center gap-2 mb-1">
                        <strong className="text-sm">Query</strong>
                        <span className="text-xs text-muted-foreground">
                          {formatTimestamp(entry.timestamp)}
                        </span>
                      </div>
                      <div className="text-left">{entry.query}</div>
                    </div>
                  </div>

                  {/* Vector Store Response */}
                  <div className="text-left">
                    <div className="inline-block max-w-[80%] rounded-lg shadow-sm p-3 bg-background border border-border">
                      <div className="flex items-center gap-2 mb-2">
                        {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                        <Database className="h-4 w-4 text-emerald-500" />
                        <strong className="text-sm">
                          Vector Store Results
                        </strong>
                        {entry.response && (
                          <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">
                            {entry.response.data?.length || 0} results
                          </span>
                        )}
                      </div>

                      {entry.response &&
                      entry.response.data &&
                      entry.response.data.length > 0 ? (
                        <div className="space-y-3">
                          {entry.response.data.map((result, resultIndex) => {
                            const isExpanded =
                              expandedResults[`${index}-${resultIndex}`] ||
                              false;

                            return (
                              <div
                                key={resultIndex}
                                className="border border-border rounded-lg overflow-hidden bg-muted"
                              >
                                {/* Clickable Header */}
                                <div
                                  className="flex justify-between items-center p-3 cursor-pointer hover:bg-muted/70 transition-colors"
                                  onClick={() =>
                                    toggleResultExpansion(index, resultIndex)
                                  }
                                >
                                  <div className="flex items-center">
                                    {isExpanded ? (
                                      <ChevronDown className="h-4 w-4 text-muted-foreground mr-2" />
                                    ) : (
                                      <ChevronRight className="h-4 w-4 text-muted-foreground mr-2" />
                                    )}
                                    <span className="font-medium text-sm">
                                      Result {resultIndex + 1}
                                    </span>
                                    {!isExpanded &&
                                      result.content &&
                                      result.content[0] && (
                                        <span className="ml-2 text-xs text-muted-foreground truncate max-w-md">
                                          -{" "}
                                          {result.content[0].text.substring(
                                            0,
                                            100,
                                          )}
                                          ...
                                        </span>
                                      )}
                                  </div>
                                  {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
                                  <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded dark:bg-blue-950/40 dark:text-blue-200">
                                    Score: {result.score.toFixed(4)}
                                  </span>
                                </div>

                                {isExpanded && (
                                  <div className="border-t border-border bg-background p-3">
                                    {result.content &&
                                      result.content.map(
                                        (content, contentIndex) => (
                                          <div
                                            key={contentIndex}
                                            className="mb-3"
                                          >
                                            <div className="text-xs text-muted-foreground mb-1">
                                              Content ({content.type})
                                            </div>
                                            <div className="text-sm bg-muted p-3 rounded border border-border max-h-40 overflow-y-auto">
                                              {content.text}
                                            </div>
                                          </div>
                                        ),
                                      )}

                                    {(result.file_id ||
                                      result.filename ||
                                      result.attributes) && (
                                      <div className="mt-3 pt-3 border-t border-border">
                                        <div className="text-xs text-muted-foreground mb-2 font-medium">
                                          Metadata
                                        </div>
                                        <div className="space-y-2 text-xs">
                                          {result.file_id && (
                                            <div className="bg-muted p-2 rounded">
                                              <span className="font-medium">
                                                File ID:
                                              </span>{" "}
                                              {result.file_id}
                                            </div>
                                          )}
                                          {result.filename && (
                                            <div className="bg-muted p-2 rounded">
                                              <span className="font-medium">
                                                Filename:
                                              </span>{" "}
                                              {result.filename}
                                            </div>
                                          )}
                                          {result.attributes &&
                                            Object.keys(result.attributes)
                                              .length > 0 && (
                                              <div className="bg-muted p-2 rounded">
                                                <span className="font-medium block mb-1">
                                                  Attributes:
                                                </span>
                                                <pre className="text-xs bg-background p-2 rounded border border-border overflow-x-auto">
                                                  {JSON.stringify(
                                                    result.attributes,
                                                    null,
                                                    2,
                                                  )}
                                                </pre>
                                              </div>
                                            )}
                                        </div>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="text-muted-foreground text-sm">
                          No results found
                        </div>
                      )}
                    </div>
                  </div>

                  {index < searchHistory.length - 1 && (
                    <hr className="border-border" />
                  )}
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="flex justify-center items-center my-4">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-border bg-background">
          <div className="flex items-end gap-2">
            <div className="flex-1">
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your search query... (Shift+Enter for new line)"
                disabled={isLoading}
                rows={1}
                className="resize-none"
              />
            </div>
            <Button
              type="button"
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
              Search
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default VectorStoreTester;
