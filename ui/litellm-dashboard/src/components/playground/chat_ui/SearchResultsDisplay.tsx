import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  ChevronDown,
  ChevronRight,
  Database,
  FileText,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { VectorStoreSearchResponse } from "./types";

interface SearchResultsDisplayProps {
  searchResults: VectorStoreSearchResponse[];
}

export function SearchResultsDisplay({
  searchResults,
}: SearchResultsDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [expandedResults, setExpandedResults] = useState<
    Record<string, boolean>
  >({});

  if (!searchResults || searchResults.length === 0) {
    return null;
  }

  const toggleResult = (pageIndex: number, resultIndex: number) => {
    const key = `${pageIndex}-${resultIndex}`;
    setExpandedResults((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const totalResults = searchResults.reduce(
    (sum, page) => sum + page.data.length,
    0,
  );

  return (
    <div className="search-results-content mt-1 mb-2">
      <Button
        variant="ghost"
        size="sm"
        className="flex items-center text-xs text-muted-foreground hover:text-foreground"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Database className="h-3.5 w-3.5" />
        {isExpanded ? "Hide sources" : `Show sources (${totalResults})`}
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 ml-1" />
        ) : (
          <ChevronRight className="h-3 w-3 ml-1" />
        )}
      </Button>

      {isExpanded && (
        <div className="mt-2 p-3 bg-muted border border-border rounded-md text-sm">
          <div className="space-y-3">
            {searchResults.map((resultPage, pageIndex) => (
              <div key={pageIndex}>
                <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                  <span className="font-medium">Query:</span>
                  <span className="italic">
                    &quot;{resultPage.search_query}&quot;
                  </span>
                  <span>•</span>
                  <span>
                    {resultPage.data.length} result
                    {resultPage.data.length !== 1 ? "s" : ""}
                  </span>
                </div>

                <div className="space-y-2">
                  {resultPage.data.map((result, resultIndex) => {
                    const isResultExpanded =
                      expandedResults[`${pageIndex}-${resultIndex}`] || false;

                    return (
                      <div
                        key={resultIndex}
                        className="border border-border rounded-md overflow-hidden bg-background"
                      >
                        <div
                          className="flex items-center justify-between p-2 cursor-pointer hover:bg-muted transition-colors"
                          onClick={() => toggleResult(pageIndex, resultIndex)}
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <ChevronRight
                              className={cn(
                                "w-4 h-4 text-muted-foreground transition-transform flex-shrink-0",
                                isResultExpanded && "rotate-90",
                              )}
                            />
                            <FileText className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                            <span className="text-xs font-medium text-foreground truncate">
                              {result.filename ||
                                result.file_id ||
                                `Result ${resultIndex + 1}`}
                            </span>
                            <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 text-xs font-mono flex-shrink-0">
                              {result.score.toFixed(3)}
                            </Badge>
                          </div>
                        </div>

                        {isResultExpanded && (
                          <div className="border-t border-border bg-background">
                            <div className="p-3 space-y-2">
                              {result.content.map((content, contentIndex) => (
                                <div key={contentIndex}>
                                  <div className="text-xs font-mono bg-muted p-2 rounded text-foreground whitespace-pre-wrap break-words">
                                    {content.text}
                                  </div>
                                </div>
                              ))}

                              {result.attributes &&
                                Object.keys(result.attributes).length > 0 && (
                                  <div className="mt-2 pt-2 border-t border-border">
                                    <div className="text-xs text-muted-foreground mb-1 font-medium">
                                      Metadata:
                                    </div>
                                    <div className="space-y-1">
                                      {Object.entries(result.attributes).map(
                                        ([key, value]) => (
                                          <div
                                            key={key}
                                            className="text-xs flex gap-2"
                                          >
                                            <span className="text-muted-foreground font-medium">
                                              {key}:
                                            </span>
                                            <span className="text-foreground font-mono break-all">
                                              {String(value)}
                                            </span>
                                          </div>
                                        ),
                                      )}
                                    </div>
                                  </div>
                                )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
