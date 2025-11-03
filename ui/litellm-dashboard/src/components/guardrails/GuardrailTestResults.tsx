import React, { useState } from "react";
import { Button, Card } from "@tremor/react";
import { Typography } from "antd";
import { CopyOutlined, CheckCircleOutlined, ClockCircleOutlined, DownOutlined, RightOutlined } from "@ant-design/icons";
import NotificationsManager from "../molecules/notifications_manager";

const { Text } = Typography;

interface TestResult {
  guardrailName: string;
  response_text: string;
  latency: number;
}

interface TestError {
  guardrailName: string;
  error: Error;
  latency: number;
}

interface GuardrailTestResultsProps {
  results: TestResult[] | null;
  errors: TestError[] | null;
}

export function GuardrailTestResults({ results, errors }: GuardrailTestResultsProps) {
  const [collapsedResults, setCollapsedResults] = useState<Set<string>>(new Set());

  const toggleResultCollapse = (guardrailName: string) => {
    const newCollapsed = new Set(collapsedResults);
    if (newCollapsed.has(guardrailName)) {
      newCollapsed.delete(guardrailName);
    } else {
      newCollapsed.add(guardrailName);
    }
    setCollapsedResults(newCollapsed);
  };

  const copyToClipboard = async (text: string) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        textArea.style.position = "fixed";
        textArea.style.opacity = "0";
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();

        const successful = document.execCommand("copy");
        document.body.removeChild(textArea);

        if (!successful) {
          throw new Error("execCommand failed");
        }
        return true;
      }
    } catch (error) {
      console.error("Copy failed:", error);
      return false;
    }
  };

  if (!results && !errors) {
    return null;
  }

  return (
    <div className="space-y-3 pt-4 border-t border-gray-200">
      <h3 className="text-sm font-semibold text-gray-900">Results</h3>

      {/* Success Results */}
      {results &&
        results.map((result) => {
          const isCollapsed = collapsedResults.has(result.guardrailName);
          return (
            <Card key={result.guardrailName} className="bg-green-50 border-green-200">
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div 
                    className="flex items-center space-x-2 cursor-pointer flex-1"
                    onClick={() => toggleResultCollapse(result.guardrailName)}
                  >
                    {isCollapsed ? (
                      <RightOutlined className="text-gray-500 text-xs" />
                    ) : (
                      <DownOutlined className="text-gray-500 text-xs" />
                    )}
                    <CheckCircleOutlined className="text-green-600 text-lg" />
                    <span className="text-sm font-medium text-green-800">
                      {result.guardrailName}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center space-x-1 text-xs text-gray-600">
                      <ClockCircleOutlined />
                      <span className="font-medium">{result.latency}ms</span>
                    </div>
                    {!isCollapsed && (
                      <Button
                        size="xs"
                        variant="secondary"
                        icon={CopyOutlined}
                        onClick={async () => {
                          const success = await copyToClipboard(result.response_text);
                          if (success) {
                            NotificationsManager.success("Result copied to clipboard");
                          } else {
                            NotificationsManager.fromBackend("Failed to copy result");
                          }
                        }}
                      >
                        Copy
                      </Button>
                    )}
                  </div>
                </div>
                {!isCollapsed && (
                  <>
                    <div className="bg-white border border-green-200 rounded p-3">
                      <label className="text-xs font-medium text-gray-600 mb-2 block">
                        Output Text
                      </label>
                      <div className="font-mono text-sm text-gray-900 whitespace-pre-wrap break-words">
                        {result.response_text}
                      </div>
                    </div>
                    <div className="text-xs text-gray-600">
                      <span className="font-medium">Characters:</span> {result.response_text.length}
                    </div>
                  </>
                )}
              </div>
            </Card>
          );
        })}

      {/* Error Results */}
      {errors &&
        errors.map((errorItem) => {
          const isCollapsed = collapsedResults.has(errorItem.guardrailName);
          return (
            <Card key={errorItem.guardrailName} className="bg-red-50 border-red-200">
              <div className="flex items-start space-x-2">
                <div 
                  className="cursor-pointer mt-0.5"
                  onClick={() => toggleResultCollapse(errorItem.guardrailName)}
                >
                  {isCollapsed ? (
                    <RightOutlined className="text-gray-500 text-xs" />
                  ) : (
                    <DownOutlined className="text-gray-500 text-xs" />
                  )}
                </div>
                <div className="text-red-600 mt-0.5">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <p 
                      className="text-sm font-medium text-red-800 cursor-pointer"
                      onClick={() => toggleResultCollapse(errorItem.guardrailName)}
                    >
                      {errorItem.guardrailName} - Error
                    </p>
                    <div className="flex items-center space-x-1 text-xs text-gray-600">
                      <ClockCircleOutlined />
                      <span className="font-medium">{errorItem.latency}ms</span>
                    </div>
                  </div>
                  {!isCollapsed && (
                    <p className="text-sm text-red-700 mt-1">{errorItem.error.message}</p>
                  )}
                </div>
              </div>
            </Card>
          );
        })}
    </div>
  );
}

export default GuardrailTestResults;

