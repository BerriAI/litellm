import React, { useState } from "react";
import { Button } from "antd";
import { VectorStoreSearchResponse } from "./types";
import { DatabaseOutlined, FileTextOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";

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
      <Button
        type="text"
        className="flex items-center text-xs text-gray-500 hover:text-gray-700"
        onClick={() => setIsExpanded(!isExpanded)}
        icon={<DatabaseOutlined />}
      >
        {isExpanded ? "Hide sources" : `Show sources (${totalResults})`}
        {isExpanded ? <DownOutlined className="ml-1" /> : <RightOutlined className="ml-1" />}
      </Button>

      {isExpanded && (
        <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md text-sm">
          <div className="space-y-3">
            {searchResults.map((resultPage, pageIndex) => (
              <div key={pageIndex}>
                <div className="text-xs text-gray-600 mb-2 flex items-center gap-2">
                  <span className="font-medium">Query:</span>
                  <span className="italic">&quot;{resultPage.search_query}&quot;</span>
                  <span className="text-gray-400">•</span>
                  <span className="text-gray-500">{resultPage.data.length} result{resultPage.data.length !== 1 ? 's' : ''}</span>
                </div>

                <div className="space-y-2">
                  {resultPage.data.map((result, resultIndex) => {
                    const isResultExpanded = expandedResults[`${pageIndex}-${resultIndex}`] || false;

                    return (
                      <div key={resultIndex} className="border border-gray-200 rounded-md overflow-hidden bg-white">
                        <div
                          className="flex items-center justify-between p-2 cursor-pointer hover:bg-gray-50 transition-colors"
                          onClick={() => toggleResult(pageIndex, resultIndex)}
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <svg
                              className={`w-4 h-4 text-gray-400 transition-transform flex-shrink-0 ${isResultExpanded ? "transform rotate-90" : ""}`}
                              fill="none"
                              stroke="currentColor"
                              viewBox="0 0 24 24"
                            >
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                            <FileTextOutlined className="text-gray-400 flex-shrink-0" style={{ fontSize: "12px" }} />
                            <span className="text-xs font-medium text-gray-700 truncate">
                              {result.filename || result.file_id || `Result ${resultIndex + 1}`}
                            </span>
                            <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700 font-mono flex-shrink-0">
                              {result.score.toFixed(3)}
                            </span>
                          </div>
                        </div>

                        {isResultExpanded && (
                          <div className="border-t border-gray-200 bg-white">
                            <div className="p-3 space-y-2">
                              {result.content.map((content, contentIndex) => (
                                <div key={contentIndex}>
                                  <div className="text-xs font-mono bg-gray-50 p-2 rounded text-gray-800 whitespace-pre-wrap break-words">
                                    {content.text}
                                  </div>
                                </div>
                              ))}

                              {result.attributes && Object.keys(result.attributes).length > 0 && (
                                <div className="mt-2 pt-2 border-t border-gray-100">
                                  <div className="text-xs text-gray-500 mb-1 font-medium">Metadata:</div>
                                  <div className="space-y-1">
                                    {Object.entries(result.attributes).map(([key, value]) => (
                                      <div key={key} className="text-xs flex gap-2">
                                        <span className="text-gray-500 font-medium">{key}:</span>
                                        <span className="text-gray-700 font-mono break-all">{String(value)}</span>
                                      </div>
                                    ))}
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

