import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { getProviderLogoAndName } from "../provider_info_helpers";

interface VectorStoreContent {
  text: string;
  type: string;
}

interface VectorStoreResult {
  score: number;
  content: VectorStoreContent[];
}

interface VectorStoreSearchResponse {
  data: VectorStoreResult[];
  search_query: string;
}

interface VectorStoreRequestMetadata {
  query: string;
  end_time: number;
  start_time: number;
  vector_store_id: string;
  custom_llm_provider: string;
  vector_store_search_response: VectorStoreSearchResponse;
}

interface VectorStoreViewerProps {
  data: VectorStoreRequestMetadata[];
}

export function VectorStoreViewer({ data }: VectorStoreViewerProps) {
  const [open, setOpen] = useState(true);
  const [expandedResults, setExpandedResults] = useState<Record<string, boolean>>({});

  if (!data || data.length === 0) {
    return null;
  }

  const formatTime = (timestamp: number): string => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const calculateDuration = (start: number, end: number): string => {
    const duration = (end - start) * 1000; // convert to milliseconds
    return `${duration.toFixed(2)}ms`;
  };

  const toggleResult = (index: number, resultIndex: number) => {
    const key = `${index}-${resultIndex}`;
    setExpandedResults((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  return (
    <div className="bg-card rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
      <Collapsible open={open} onOpenChange={setOpen}>
        <CollapsibleTrigger className="flex w-full items-center gap-3 px-4 py-3 text-left">
          {open ? (
            <ChevronDown className="size-3.5 shrink-0 text-muted-foreground" />
          ) : (
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground" />
          )}
          <h3 className="text-lg font-medium text-foreground">Vector Store Requests</h3>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <div className="p-4">
            {data.map((request, index) => (
              <div key={index} className="mb-6 last:mb-0">
                <div className="bg-card rounded-lg border p-4 mb-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="flex">
                        <span className="font-medium w-1/3">Query:</span>
                        <span className="font-mono">{request.query}</span>
                      </div>
                      <div className="flex">
                        <span className="font-medium w-1/3">Vector Store ID:</span>
                        <span className="font-mono">{request.vector_store_id}</span>
                      </div>
                      <div className="flex">
                        <span className="font-medium w-1/3">Provider:</span>
                        <span className="flex items-center">
                          {(() => {
                            const { logo, displayName } = getProviderLogoAndName(request.custom_llm_provider);
                            return (
                              <>
                                {logo && <img src={logo} alt={`${displayName} logo`} className="h-5 w-5 mr-2" />}
                                {displayName}
                              </>
                            );
                          })()}
                        </span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex">
                        <span className="font-medium w-1/3">Start Time:</span>
                        <span>{formatTime(request.start_time)}</span>
                      </div>
                      <div className="flex">
                        <span className="font-medium w-1/3">End Time:</span>
                        <span>{formatTime(request.end_time)}</span>
                      </div>
                      <div className="flex">
                        <span className="font-medium w-1/3">Duration:</span>
                        <span>{calculateDuration(request.start_time, request.end_time)}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <h4 className="font-medium mb-2">Search Results</h4>
                <div className="space-y-2">
                  {request.vector_store_search_response.data.map((result, resultIndex) => {
                    const isExpanded = expandedResults[`${index}-${resultIndex}`] || false;

                    return (
                      <div key={resultIndex} className="border rounded-lg overflow-hidden">
                        <div
                          className="flex items-center p-3 bg-muted cursor-pointer"
                          onClick={() => toggleResult(index, resultIndex)}
                        >
                          <svg
                            className={`w-5 h-5 mr-2 transition-transform ${isExpanded ? "transform rotate-90" : ""}`}
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                          <div className="flex items-center">
                            <span className="font-medium mr-2">Result {resultIndex + 1}</span>
                            <span className="text-muted-foreground text-sm">
                              Score: <span className="font-mono">{result.score.toFixed(4)}</span>
                            </span>
                          </div>
                        </div>

                        {isExpanded && (
                          <div className="p-3 border-t bg-card">
                            {result.content.map((content, contentIndex) => (
                              <div key={contentIndex} className="mb-2 last:mb-0">
                                <div className="text-xs text-muted-foreground mb-1">{content.type}</div>
                                <pre className="text-xs font-mono whitespace-pre-wrap break-all bg-muted p-2 rounded-sm">
                                  {content.text}
                                </pre>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}
