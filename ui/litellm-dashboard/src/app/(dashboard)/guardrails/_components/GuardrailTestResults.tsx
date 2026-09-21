import React, { useState } from "react";
import { Check, ChevronDown, ChevronRight, Clock, Copy } from "lucide-react";
import { toast } from "@/lib/toast";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

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
    <div className="space-y-3 border-t border-border pt-4">
      <h3 className="text-sm font-semibold">Results</h3>

      {/* Success Results */}
      {results &&
        results.map((result) => {
          const isCollapsed = collapsedResults.has(result.guardrailName);
          return (
            <Card key={result.guardrailName} className="border-success/20 bg-success/10">
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <div
                    className="flex flex-1 cursor-pointer items-center space-x-2"
                    onClick={() => toggleResultCollapse(result.guardrailName)}
                  >
                    {isCollapsed ? (
                      <ChevronRight className="size-3 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="size-3 text-muted-foreground" />
                    )}
                    <Check className="size-4 text-success" />
                    <span className="text-sm font-medium text-success">{result.guardrailName}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                      <Clock className="size-3" />
                      <span className="font-medium">{result.latency}ms</span>
                    </div>
                    {!isCollapsed && (
                      <Button
                        size="sm"
                        variant="secondary"
                        onClick={async () => {
                          const success = await copyToClipboard(result.response_text);
                          if (success) {
                            toast.success("Result copied to clipboard");
                          } else {
                            toast.fromError("Failed to copy result");
                          }
                        }}
                      >
                        <Copy />
                        Copy
                      </Button>
                    )}
                  </div>
                </div>
                {!isCollapsed && (
                  <>
                    <div className="rounded-sm border border-success/20 bg-background p-3">
                      <label className="mb-2 block text-xs font-medium text-muted-foreground">Output Text</label>
                      <div className="font-mono text-sm whitespace-pre-wrap wrap-break-word">
                        {result.response_text}
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground">
                      <span className="font-medium">Characters:</span> {result.response_text.length}
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          );
        })}

      {/* Error Results */}
      {errors &&
        errors.map((errorItem) => {
          const isCollapsed = collapsedResults.has(errorItem.guardrailName);
          return (
            <Card key={errorItem.guardrailName} className="border-destructive/20 bg-destructive/10">
              <CardContent>
                <div className="flex items-start space-x-2">
                  <div className="mt-0.5 cursor-pointer" onClick={() => toggleResultCollapse(errorItem.guardrailName)}>
                    {isCollapsed ? (
                      <ChevronRight className="size-3 text-muted-foreground" />
                    ) : (
                      <ChevronDown className="size-3 text-muted-foreground" />
                    )}
                  </div>
                  <div className="mt-0.5 text-destructive">
                    <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                      <path
                        fillRule="evenodd"
                        d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                        clipRule="evenodd"
                      />
                    </svg>
                  </div>
                  <div className="flex-1">
                    <div className="mb-1 flex items-center justify-between">
                      <p
                        className="cursor-pointer text-sm font-medium text-destructive"
                        onClick={() => toggleResultCollapse(errorItem.guardrailName)}
                      >
                        {errorItem.guardrailName} - Error
                      </p>
                      <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                        <Clock className="size-3" />
                        <span className="font-medium">{errorItem.latency}ms</span>
                      </div>
                    </div>
                    {!isCollapsed && <p className="mt-1 text-sm text-destructive">{errorItem.error.message}</p>}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
    </div>
  );
}

export default GuardrailTestResults;
