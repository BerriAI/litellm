import React, { useState } from "react";
import MessageManager from "@/components/molecules/message_manager";
import { ChevronDown, ChevronRight, Database, Send } from "lucide-react";
import { vectorStoreSearchCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";

interface VectorStoreContent {
  text: string;
  type: string;
}

interface VectorStoreResult {
  score: number;
  content: VectorStoreContent[];
  file_id?: string;
  filename?: string;
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

export const VectorStoreTester: React.FC<VectorStoreTesterProps> = ({ vectorStoreId, accessToken, className = "" }) => {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [searchHistory, setSearchHistory] = useState<
    {
      query: string;
      response: VectorStoreSearchResponse | null;
      timestamp: number;
    }[]
  >([]);
  const [expandedResults, setExpandedResults] = useState<Record<string, boolean>>({});

  const handleSearch = async () => {
    if (!query.trim()) {
      MessageManager.warning("Please enter a search query");
      return;
    }

    setIsLoading(true);

    try {
      const response = await vectorStoreSearchCall(accessToken, vectorStoreId, query);

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
    <Card className={`w-full py-0 shadow-md ${className}`}>
      <div className="flex h-150 flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b p-4">
          <div className="flex items-center">
            <Database className="mr-2 size-4 text-primary" />
            <h4 className="text-base font-medium text-foreground">Test Vector Store</h4>
          </div>
          {searchHistory.length > 0 && (
            <Button variant="outline" size="sm" onClick={clearHistory}>
              Clear History
            </Button>
          )}
        </div>

        {/* Results Area */}
        <div className="flex-1 overflow-auto p-4 pb-0">
          {searchHistory.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center text-muted-foreground">
              <Database className="mb-4 size-12" />
              <p className="text-sm">Test your vector store by entering a search query below</p>
            </div>
          ) : (
            <div className="space-y-4">
              {searchHistory.map((entry, index) => (
                <div key={index} className="space-y-2">
                  {/* User Query */}
                  <div className="text-right">
                    <div className="inline-block max-w-[80%] rounded-lg bg-muted p-3 shadow-xs ring-1 ring-foreground/10">
                      <div className="mb-1 flex items-center gap-2">
                        <strong className="text-sm">Query</strong>
                        <span className="text-xs text-muted-foreground">{formatTimestamp(entry.timestamp)}</span>
                      </div>
                      <div className="text-left">{entry.query}</div>
                    </div>
                  </div>

                  {/* Vector Store Response */}
                  <div className="text-left">
                    <div className="inline-block max-w-[80%] rounded-lg bg-card p-3 shadow-xs ring-1 ring-foreground/10">
                      <div className="mb-2 flex items-center gap-2">
                        <Database className="size-4 text-primary" />
                        <strong className="text-sm">Vector Store Results</strong>
                        {entry.response && (
                          <span className="rounded-sm bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                            {entry.response.data?.length || 0} results
                          </span>
                        )}
                      </div>

                      {entry.response && entry.response.data && entry.response.data.length > 0 ? (
                        <div className="space-y-3">
                          {entry.response.data.map((result, resultIndex) => {
                            const isExpanded = expandedResults[`${index}-${resultIndex}`] || false;

                            return (
                              <div key={resultIndex} className="overflow-hidden rounded-lg border bg-muted/50">
                                {/* Clickable Header */}
                                <div
                                  className="flex cursor-pointer items-center justify-between p-3 transition-colors hover:bg-muted"
                                  onClick={() => toggleResultExpansion(index, resultIndex)}
                                >
                                  <div className="flex items-center">
                                    {isExpanded ? (
                                      <ChevronDown className="mr-2 size-4 text-muted-foreground" />
                                    ) : (
                                      <ChevronRight className="mr-2 size-4 text-muted-foreground" />
                                    )}
                                    <span className="text-sm font-medium">Result {resultIndex + 1}</span>
                                    {/* Show preview of content when collapsed */}
                                    {!isExpanded && result.content && result.content[0] && (
                                      <span className="ml-2 max-w-md truncate text-xs text-muted-foreground">
                                        - {result.content[0].text.substring(0, 100)}...
                                      </span>
                                    )}
                                  </div>
                                  <span className="rounded-sm bg-muted px-2 py-1 text-xs text-foreground">
                                    Score: {result.score.toFixed(4)}
                                  </span>
                                </div>

                                {/* Expandable Content */}
                                {isExpanded && (
                                  <div className="border-t bg-card p-3">
                                    {/* Content */}
                                    {result.content &&
                                      result.content.map((content, contentIndex) => (
                                        <div key={contentIndex} className="mb-3">
                                          <div className="mb-1 text-xs text-muted-foreground">
                                            Content ({content.type})
                                          </div>
                                          <div className="max-h-40 overflow-y-auto rounded-sm border bg-muted/50 p-3 text-sm text-foreground">
                                            {content.text}
                                          </div>
                                        </div>
                                      ))}

                                    {/* Metadata */}
                                    {(result.file_id || result.filename || result.attributes) && (
                                      <div className="mt-3 border-t pt-3">
                                        <div className="mb-2 text-xs font-medium text-muted-foreground">Metadata</div>
                                        <div className="space-y-2 text-xs">
                                          {result.file_id && (
                                            <div className="rounded-sm bg-muted/50 p-2">
                                              <span className="font-medium">File ID:</span> {result.file_id}
                                            </div>
                                          )}
                                          {result.filename && (
                                            <div className="rounded-sm bg-muted/50 p-2">
                                              <span className="font-medium">Filename:</span> {result.filename}
                                            </div>
                                          )}
                                          {result.attributes && Object.keys(result.attributes).length > 0 && (
                                            <div className="rounded-sm bg-muted/50 p-2">
                                              <span className="mb-1 block font-medium">Attributes:</span>
                                              <pre className="overflow-x-auto rounded-sm border bg-card p-2 text-xs">
                                                {JSON.stringify(result.attributes, null, 2)}
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
                        <div className="text-sm text-muted-foreground">No results found</div>
                      )}
                    </div>
                  </div>

                  {index < searchHistory.length - 1 && <Separator />}
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="my-4 flex items-center justify-center">
              <UiLoadingSpinner className="size-6 text-primary" />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="border-t bg-card p-4">
          <div className="flex items-end space-x-2">
            <div className="flex-1">
              <Textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your search query... (Shift+Enter for new line)"
                disabled={isLoading}
                rows={1}
                className="field-sizing-fixed max-h-24 min-h-9 resize-none"
              />
            </div>
            <Button onClick={handleSearch} disabled={isLoading || !query.trim()}>
              {isLoading ? <UiLoadingSpinner className="size-4" /> : <Send className="size-4" />}
              Search
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default VectorStoreTester;
