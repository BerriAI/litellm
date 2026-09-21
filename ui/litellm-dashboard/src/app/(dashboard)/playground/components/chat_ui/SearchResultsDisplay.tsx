import React, { useState } from "react";
import { VectorStoreSearchResponse } from "@/components/chat_ui/types";
import { ChevronDown, ChevronRight, Database, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface SearchResultsDisplayProps {
  searchResults: VectorStoreSearchResponse[];
}

export function SearchResultsDisplay({ searchResults }: SearchResultsDisplayProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [expandedResults, setExpandedResults] = useState<Record<string, boolean>>({});

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

  const totalResults = searchResults.reduce((sum, page) => sum + page.data.length, 0);

  return (
    <div className="search-results-content mt-1 mb-2">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-xs text-muted-foreground hover:text-foreground"
            />
          }
        >
          <Database className="size-4" />
          {isExpanded ? "Hide sources" : `Show sources (${totalResults})`}
          {isExpanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div className="mt-2 p-3 bg-muted border border-border rounded-md text-sm">
            <div className="space-y-3">
              {searchResults.map((resultPage, pageIndex) => (
                <div key={pageIndex}>
                  <div className="text-xs text-muted-foreground mb-2 flex items-center gap-2">
                    <span className="font-medium">Query:</span>
                    <span className="italic">&quot;{resultPage.search_query}&quot;</span>
                    <span className="text-muted-foreground">•</span>
                    <span className="text-muted-foreground">
                      {resultPage.data.length} result{resultPage.data.length !== 1 ? "s" : ""}
                    </span>
                  </div>

                  <div className="space-y-2">
                    {resultPage.data.map((result, resultIndex) => {
                      const isResultExpanded = expandedResults[`${pageIndex}-${resultIndex}`] || false;

                      return (
                        <Collapsible
                          key={resultIndex}
                          open={isResultExpanded}
                          onOpenChange={() => toggleResult(pageIndex, resultIndex)}
                          className="overflow-hidden rounded-md border border-border bg-card"
                        >
                          <CollapsibleTrigger className="flex w-full items-center justify-between p-2 text-left transition-colors hover:bg-accent">
                            <div className="flex items-center gap-2 flex-1 min-w-0">
                              <ChevronRight
                                className={`size-4 shrink-0 text-muted-foreground transition-transform ${isResultExpanded ? "rotate-90" : ""}`}
                              />
                              <FileText className="size-3 shrink-0 text-muted-foreground" />
                              <span className="text-xs font-medium text-foreground truncate">
                                {result.filename || result.file_id || `Result ${resultIndex + 1}`}
                              </span>
                              <span className="text-xs px-2 py-0.5 rounded-sm bg-info/15 text-info font-mono shrink-0">
                                {result.score.toFixed(3)}
                              </span>
                            </div>
                          </CollapsibleTrigger>

                          <CollapsibleContent>
                            <div className="border-t border-border bg-card">
                              <div className="p-3 space-y-2">
                                {result.content.map((content, contentIndex) => (
                                  <div key={contentIndex}>
                                    <div className="text-xs font-mono bg-muted p-2 rounded-sm text-foreground whitespace-pre-wrap wrap-break-word">
                                      {content.text}
                                    </div>
                                  </div>
                                ))}

                                {result.attributes && Object.keys(result.attributes).length > 0 && (
                                  <div className="mt-2 pt-2 border-t border-border">
                                    <div className="text-xs text-muted-foreground mb-1 font-medium">Metadata:</div>
                                    <div className="space-y-1">
                                      {Object.entries(result.attributes).map(([key, value]) => (
                                        <div key={key} className="text-xs flex gap-2">
                                          <span className="text-muted-foreground font-medium">{key}:</span>
                                          <span className="text-foreground font-mono break-all">{String(value)}</span>
                                        </div>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </CollapsibleContent>
                        </Collapsible>
                      );
                    })}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
