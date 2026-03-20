import { InfoCircleOutlined, WarningOutlined } from "@ant-design/icons";
import { Button, Divider, Typography } from "antd";
import React, { useEffect, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { testSearchToolConnection } from "../networking";

const { Text } = Typography;

interface SearchConnectionTestProps {
  litellmParams: Record<string, any>;
  accessToken: string;
  onTestComplete?: () => void;
}

const SearchConnectionTest: React.FC<SearchConnectionTestProps> = ({
  litellmParams,
  accessToken,
  onTestComplete,
}) => {
  const [isLoading, setIsLoading] = useState(true);
  const [testResult, setTestResult] = useState<{
    status: "success" | "error";
    message: string;
    test_query?: string;
    results_count?: number;
    error_type?: string;
  } | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    const runTest = async () => {
      setIsLoading(true);
      try {
        const result = await testSearchToolConnection(accessToken, litellmParams);
        setTestResult(result);
        if (result.status === "success") {
          NotificationsManager.success("Connection test successful!");
        }
      } catch (error) {
        setTestResult({
          status: "error",
          message: error instanceof Error ? error.message : "Unknown error occurred",
          error_type: "NetworkError",
        });
      } finally {
        setIsLoading(false);
        if (onTestComplete) {
          onTestComplete();
        }
      }
    };

    runTest();
  }, [accessToken, litellmParams, onTestComplete]);

  const getCleanErrorMessage = (errorMsg: string) => {
    if (!errorMsg) return "Unknown error";

    // Remove stack traces
    const mainError = errorMsg.split("stack trace:")[0].trim();

    // Remove litellm error prefixes
    const cleanedError = mainError.replace(/^litellm\.(.*?)Error:\s*/, "");

    // Remove AuthenticationError prefix if it exists
    const finalError = cleanedError.replace(/^AuthenticationError:\s*/, "");

    // If the error contains HTML (like a 401 page), extract just the key info
    if (finalError.includes("<html>") || finalError.includes("<!DOCTYPE")) {
      // Try to extract the title or main error from HTML
      const titleMatch = finalError.match(/<title>(.*?)<\/title>/);
      if (titleMatch) {
        return titleMatch[1];
      }
      // If it's a 401 error
      if (finalError.includes("401") || finalError.includes("Authorization Required")) {
        return "Authentication failed: Invalid API key or credentials";
      }
      return "Authentication error - please check your API key";
    }

    // Limit very long error messages
    if (finalError.length > 200) {
      return finalError.substring(0, 200) + "...";
    }

    return finalError;
  };

  const errorMessage = testResult?.message ? getCleanErrorMessage(testResult.message) : "Unknown error";

  if (isLoading) {
    return (
      <div style={{ padding: "24px", borderRadius: "8px", backgroundColor: "#fff" }}>
        <div style={{ textAlign: "center", padding: "32px 20px" }}>
          <div className="loading-spinner" style={{ marginBottom: "16px" }}>
            <div
              style={{
                border: "3px solid #f3f3f3",
                borderTop: "3px solid #1890ff",
                borderRadius: "50%",
                width: "30px",
                height: "30px",
                animation: "spin 1s linear infinite",
                margin: "0 auto",
              }}
            />
          </div>
          <Text style={{ fontSize: "16px" }}>
            Testing connection to {litellmParams.search_provider || "search provider"}...
          </Text>
          <style jsx>{`
            @keyframes spin {
              0% {
                transform: rotate(0deg);
              }
              100% {
                transform: rotate(360deg);
              }
            }
          `}</style>
        </div>
      </div>
    );
  }

  if (!testResult) {
    return null;
  }

  return (
    <div style={{ padding: "24px", borderRadius: "8px", backgroundColor: "#fff" }}>
      {testResult.status === "success" ? (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "32px 20px" }}>
          <div style={{ color: "#52c41a", fontSize: "24px", display: "flex", alignItems: "center" }}>
            <svg
              viewBox="64 64 896 896"
              focusable="false"
              data-icon="check-circle"
              width="1em"
              height="1em"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M512 64C264.6 64 64 264.6 64 512s200.6 448 448 448 448-200.6 448-448S759.4 64 512 64zm193.5 301.7l-210.6 292a31.8 31.8 0 01-51.7 0L318.5 484.9c-3.8-5.3 0-12.7 6.5-12.7h46.9c10.2 0 19.9 4.9 25.9 13.3l71.2 98.8 157.2-218c6-8.3 15.6-13.3 25.9-13.3H699c6.5 0 10.3 7.4 6.5 12.7z"></path>
            </svg>
          </div>
          <div style={{ marginLeft: "12px" }}>
            <Text type="success" style={{ fontSize: "18px", fontWeight: 500, display: "block" }}>
              Connection to {litellmParams.search_provider} successful!
            </Text>
            {testResult.test_query && (
              <Text style={{ fontSize: "14px", color: "#666", marginTop: "8px", display: "block" }}>
                Test query: <code style={{ backgroundColor: "#f0f0f0", padding: "2px 6px", borderRadius: "4px" }}>{testResult.test_query}</code>
              </Text>
            )}
            {testResult.results_count !== undefined && (
              <Text style={{ fontSize: "14px", color: "#666", display: "block" }}>
                Results retrieved: {testResult.results_count}
              </Text>
            )}
          </div>
        </div>
      ) : (
        <>
          <div>
            <div style={{ display: "flex", alignItems: "center", marginBottom: "20px" }}>
              <WarningOutlined style={{ color: "#ff4d4f", fontSize: "24px", marginRight: "12px" }} />
              <Text type="danger" style={{ fontSize: "18px", fontWeight: 500 }}>
                Connection to {litellmParams.search_provider || "search provider"} failed
              </Text>
            </div>

            <div
              style={{
                backgroundColor: "#fff2f0",
                border: "1px solid #ffccc7",
                borderRadius: "8px",
                padding: "16px",
                marginBottom: "20px",
                boxShadow: "0 1px 2px rgba(0, 0, 0, 0.03)",
              }}
            >
              <Text strong style={{ display: "block", marginBottom: "8px" }}>
                Error:{" "}
              </Text>
              <Text type="danger" style={{ fontSize: "14px", lineHeight: "1.5" }}>
                {errorMessage}
              </Text>

              {testResult.error_type && (
                <div style={{ marginTop: "8px" }}>
                  <Text style={{ fontSize: "13px", color: "#666" }}>
                    Error type:{" "}
                    <code style={{ backgroundColor: "#ffebee", padding: "2px 6px", borderRadius: "4px", color: "#d32f2f" }}>
                      {testResult.error_type}
                    </code>
                  </Text>
                </div>
              )}

              {testResult.message && (
                <div style={{ marginTop: "12px" }}>
                  <Button
                    type="link"
                    onClick={() => setShowDetails(!showDetails)}
                    style={{ paddingLeft: 0, height: "auto" }}
                  >
                    {showDetails ? "Hide Details" : "Show Details"}
                  </Button>
                </div>
              )}
            </div>

            {showDetails && (
              <div style={{ marginBottom: "20px" }}>
                <Text strong style={{ display: "block", marginBottom: "8px", fontSize: "15px" }}>
                  Full Error Details
                </Text>
                <pre
                  style={{
                    backgroundColor: "#f5f5f5",
                    padding: "16px",
                    borderRadius: "8px",
                    fontSize: "13px",
                    maxHeight: "200px",
                    overflow: "auto",
                    border: "1px solid #e8e8e8",
                    lineHeight: "1.5",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {testResult.message}
                </pre>
              </div>
            )}

            <div
              style={{
                backgroundColor: "#fffbf0",
                border: "1px solid #ffe58f",
                borderLeft: "4px solid #faad14",
                borderRadius: "8px",
                padding: "16px",
              }}
            >
              <Text strong style={{ display: "block", marginBottom: "8px", color: "#d48806" }}>
                Troubleshooting tips:
              </Text>
              <ul style={{ margin: "8px 0", paddingLeft: "20px", color: "#ad6800" }}>
                <li style={{ marginBottom: "6px" }}>Verify your API key is correct and active</li>
                <li style={{ marginBottom: "6px" }}>Check if the search provider service is operational</li>
                <li style={{ marginBottom: "6px" }}>Ensure you have sufficient credits/quota with the provider</li>
                <li style={{ marginBottom: "6px" }}>Review the provider&apos;s documentation for any additional requirements</li>
              </ul>
            </div>
          </div>
        </>
      )}
      <Divider style={{ margin: "24px 0 16px" }} />
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Button type="link" href="https://docs.litellm.ai/docs/search" target="_blank" icon={<InfoCircleOutlined />}>
          View Search Documentation
        </Button>
      </div>
    </div>
  );
};

export default SearchConnectionTest;

