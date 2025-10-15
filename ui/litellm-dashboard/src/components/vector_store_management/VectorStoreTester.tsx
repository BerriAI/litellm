import React, { useState } from "react";
import { Button, Input, Card, Typography, Spin, message, Divider } from "antd";
import { SendOutlined, DatabaseOutlined, LoadingOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";
import { vectorStoreSearchCall } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { TextArea } = Input;
const { Text, Title } = Typography;

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
      message.warning("Please enter a search query");
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
    <Card className="w-full rounded-xl shadow-md">
      <div className="flex flex-col h-[600px]">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 flex justify-between items-center">
          <div className="flex items-center">
            <DatabaseOutlined className="mr-2 text-blue-500" />
            <Title level={4} className="mb-0">
              Test Vector Store
            </Title>
          </div>
          {searchHistory.length > 0 && (
            <Button onClick={clearHistory} size="small">
              Clear History
            </Button>
          )}
        </div>

        {/* Results Area */}
        <div className="flex-1 overflow-auto p-4 pb-0">
          {searchHistory.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <DatabaseOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
              <Text>Test your vector store by entering a search query below</Text>
            </div>
          ) : (
            <div className="space-y-4">
              {searchHistory.map((entry, index) => (
                <div key={index} className="space-y-2">
                  {/* User Query */}
                  <div className="text-right">
                    <div className="inline-block max-w-[80%] rounded-lg shadow-sm p-3 bg-blue-50 border border-blue-200">
                      <div className="flex items-center gap-2 mb-1">
                        <strong className="text-sm">Query</strong>
                        <span className="text-xs text-gray-500">{formatTimestamp(entry.timestamp)}</span>
                      </div>
                      <div className="text-left">{entry.query}</div>
                    </div>
                  </div>

                  {/* Vector Store Response */}
                  <div className="text-left">
                    <div className="inline-block max-w-[80%] rounded-lg shadow-sm p-3 bg-white border border-gray-200">
                      <div className="flex items-center gap-2 mb-2">
                        <DatabaseOutlined className="text-green-500" />
                        <strong className="text-sm">Vector Store Results</strong>
                        {entry.response && (
                          <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-600">
                            {entry.response.data?.length || 0} results
                          </span>
                        )}
                      </div>

                      {entry.response && entry.response.data && entry.response.data.length > 0 ? (
                        <div className="space-y-3">
                          {entry.response.data.map((result, resultIndex) => {
                            const isExpanded = expandedResults[`${index}-${resultIndex}`] || false;

                            return (
                              <div key={resultIndex} className="border rounded-lg overflow-hidden bg-gray-50">
                                {/* Clickable Header */}
                                <div
                                  className="flex justify-between items-center p-3 cursor-pointer hover:bg-gray-100 transition-colors"
                                  onClick={() => toggleResultExpansion(index, resultIndex)}
                                >
                                  <div className="flex items-center">
                                    {isExpanded ? (
                                      <DownOutlined className="text-gray-500 mr-2" />
                                    ) : (
                                      <RightOutlined className="text-gray-500 mr-2" />
                                    )}
                                    <span className="font-medium text-sm">Result {resultIndex + 1}</span>
                                    {/* Show preview of content when collapsed */}
                                    {!isExpanded && result.content && result.content[0] && (
                                      <span className="ml-2 text-xs text-gray-500 truncate max-w-md">
                                        - {result.content[0].text.substring(0, 100)}...
                                      </span>
                                    )}
                                  </div>
                                  <span className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                    Score: {result.score.toFixed(4)}
                                  </span>
                                </div>

                                {/* Expandable Content */}
                                {isExpanded && (
                                  <div className="border-t bg-white p-3">
                                    {/* Content */}
                                    {result.content &&
                                      result.content.map((content, contentIndex) => (
                                        <div key={contentIndex} className="mb-3">
                                          <div className="text-xs text-gray-500 mb-1">Content ({content.type})</div>
                                          <div className="text-sm bg-gray-50 p-3 rounded border text-gray-800 max-h-40 overflow-y-auto">
                                            {content.text}
                                          </div>
                                        </div>
                                      ))}

                                    {/* Metadata */}
                                    {(result.file_id || result.filename || result.attributes) && (
                                      <div className="mt-3 pt-3 border-t border-gray-200">
                                        <div className="text-xs text-gray-500 mb-2 font-medium">Metadata</div>
                                        <div className="space-y-2 text-xs">
                                          {result.file_id && (
                                            <div className="bg-gray-50 p-2 rounded">
                                              <span className="font-medium">File ID:</span> {result.file_id}
                                            </div>
                                          )}
                                          {result.filename && (
                                            <div className="bg-gray-50 p-2 rounded">
                                              <span className="font-medium">Filename:</span> {result.filename}
                                            </div>
                                          )}
                                          {result.attributes && Object.keys(result.attributes).length > 0 && (
                                            <div className="bg-gray-50 p-2 rounded">
                                              <span className="font-medium block mb-1">Attributes:</span>
                                              <pre className="text-xs bg-white p-2 rounded border overflow-x-auto">
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
                        <div className="text-gray-500 text-sm">No results found</div>
                      )}
                    </div>
                  </div>

                  {index < searchHistory.length - 1 && <Divider />}
                </div>
              ))}
            </div>
          )}

          {isLoading && (
            <div className="flex justify-center items-center my-4">
              <Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 border-t border-gray-200 bg-white">
          <div className="flex items-end space-x-2">
            <div className="flex-1">
              <TextArea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your search query... (Shift+Enter for new line)"
                disabled={isLoading}
                autoSize={{ minRows: 1, maxRows: 4 }}
                style={{ resize: "none" }}
              />
            </div>
            <Button
              type="primary"
              onClick={handleSearch}
              disabled={isLoading || !query.trim()}
              icon={<SendOutlined />}
              loading={isLoading}
            >
              Search
            </Button>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default VectorStoreTester;
