import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Copy,
  XCircle,
} from "lucide-react";
import NotificationsManager from "../molecules/notifications_manager";

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

export function GuardrailTestResults({
  results,
  errors,
}: GuardrailTestResultsProps) {
  const [collapsedResults, setCollapsedResults] = useState<Set<string>>(
    new Set(),
  );

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
    <div className="space-y-3 pt-4 border-t border-border">
      <h3 className="text-sm font-semibold text-foreground">Results</h3>

      {results &&
        results.map((result) => {
          const isCollapsed = collapsedResults.has(result.guardrailName);
          return (
            <Card
              key={result.guardrailName}
              className="bg-emerald-50 border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900 p-4"
            >
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div
                    className="flex items-center space-x-2 cursor-pointer flex-1"
                    onClick={() => toggleResultCollapse(result.guardrailName)}
                  >
                    {isCollapsed ? (
                      <ChevronRight className="h-3 w-3 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="h-3 w-3 text-muted-foreground" />
                    )}
                    <CheckCircle2 className="h-5 w-5 text-emerald-600 dark:text-emerald-400" />
                    <span className="text-sm font-medium text-emerald-800 dark:text-emerald-200">
                      {result.guardrailName}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span className="font-medium">{result.latency}ms</span>
                    </div>
                    {!isCollapsed && (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={async () => {
                          const success = await copyToClipboard(
                            result.response_text,
                          );
                          if (success) {
                            NotificationsManager.success(
                              "Result copied to clipboard",
                            );
                          } else {
                            NotificationsManager.fromBackend(
                              "Failed to copy result",
                            );
                          }
                        }}
                      >
                        <Copy className="h-3 w-3" />
                        Copy
                      </Button>
                    )}
                  </div>
                </div>
                {!isCollapsed && (
                  <>
                    <div className="bg-background border border-emerald-200 dark:border-emerald-900 rounded p-3">
                      <label className="text-xs font-medium text-muted-foreground mb-2 block">
                        Output Text
                      </label>
                      <div className="font-mono text-sm text-foreground whitespace-pre-wrap break-words">
                        {result.response_text}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      <span className="font-medium">Characters:</span>{" "}
                      {result.response_text.length}
                    </div>
                  </>
                )}
              </div>
            </Card>
          );
        })}

      {errors &&
        errors.map((errorItem) => {
          const isCollapsed = collapsedResults.has(errorItem.guardrailName);
          return (
            <Card
              key={errorItem.guardrailName}
              className="bg-red-50 border-red-200 dark:bg-red-950/30 dark:border-red-900 p-4"
            >
              <div className="flex items-start space-x-2">
                <div
                  className="cursor-pointer mt-0.5"
                  onClick={() =>
                    toggleResultCollapse(errorItem.guardrailName)
                  }
                >
                  {isCollapsed ? (
                    <ChevronRight className="h-3 w-3 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="h-3 w-3 text-muted-foreground" />
                  )}
                </div>
                <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 mt-0.5" />
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <p
                      className="text-sm font-medium text-red-800 dark:text-red-200 cursor-pointer"
                      onClick={() =>
                        toggleResultCollapse(errorItem.guardrailName)
                      }
                    >
                      {errorItem.guardrailName} - Error
                    </p>
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span className="font-medium">
                        {errorItem.latency}ms
                      </span>
                    </div>
                  </div>
                  {!isCollapsed && (
                    <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                      {errorItem.error.message}
                    </p>
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
