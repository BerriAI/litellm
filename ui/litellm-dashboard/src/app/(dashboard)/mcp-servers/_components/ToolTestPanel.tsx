import React from "react";
import { CircleHelp, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { MCPTool, InputSchema } from "@/components/mcp_tools/types";
import { resolveLogoSrc } from "@/lib/assetPaths";
import { toast } from "@/lib/toast";
import { ToolArgumentsForm } from "./ToolArgumentsForm";
import { ToolArgumentField, argumentsFormKey, hasNestedParamsSchema, toolArgumentFields } from "./toolCallArguments";

export function ToolTestPanel({
  tool,
  onSubmit,
  isLoading,
  result,
  error,
  onClose,
}: {
  tool: MCPTool;
  onSubmit: (args: Record<string, any>) => void;
  isLoading: boolean;
  result: any | null;
  error: Error | null;
  onClose: () => void;
}) {
  const [viewMode, setViewMode] = React.useState<"formatted" | "json">("formatted");
  const [startTime, setStartTime] = React.useState<number | null>(null);
  const [duration, setDuration] = React.useState<number | null>(null);

  // Create a placeholder schema if we only have the "tool_input_schema" string
  const schema: InputSchema = React.useMemo(() => {
    if (typeof tool.inputSchema === "string") {
      // Default schema with a single text field
      return {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "Input for this tool",
          },
        },
        required: ["input"],
      };
    }
    return tool.inputSchema as InputSchema;
  }, [tool.inputSchema]);

  // Check if this is a nested params structure and extract the actual parameters
  const actualSchema: InputSchema = React.useMemo(() => {
    if (
      schema.properties &&
      schema.properties.params &&
      schema.properties.params.type === "object" &&
      schema.properties.params.properties
    ) {
      // This is a nested params structure, extract the actual parameters
      return {
        type: "object",
        properties: schema.properties.params.properties,
        required: schema.properties.params.required || [],
      };
    }
    return schema;
  }, [schema]);

  const argumentFields: readonly ToolArgumentField[] = React.useMemo(
    () => toolArgumentFields(actualSchema),
    [actualSchema],
  );
  const wrapInParams = React.useMemo(() => hasNestedParamsSchema(schema), [schema]);
  const formKey = React.useMemo(() => `${tool.name}:${argumentsFormKey(actualSchema)}`, [tool.name, actualSchema]);

  const runToolCall = (args: Record<string, unknown>) => {
    setStartTime(Date.now());
    setDuration(null);
    onSubmit(wrapInParams ? { params: args } : args);
  };

  // Track when result changes to calculate duration
  React.useEffect(() => {
    if (startTime && (result || error)) {
      const endTime = Date.now();
      setDuration(endTime - startTime);
    }
  }, [result, error, startTime]);

  const copyToClipboard = async (text: string) => {
    try {
      // Try modern clipboard API first
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return true;
      } else {
        // Fallback for non-secure contexts (like 0.0.0.0)
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

  const handleCopyResult = async () => {
    const success = await copyToClipboard(JSON.stringify(result, null, 2));
    if (success) {
      toast.success("Result copied to clipboard");
    } else {
      toast.fromError("Failed to copy result");
    }
  };

  const handleCopyToolName = async () => {
    const success = await copyToClipboard(tool.name);
    if (success) {
      toast.success("Tool name copied to clipboard");
    } else {
      toast.fromError("Failed to copy tool name");
    }
  };

  return (
    <div className="space-y-4 h-full">
      {/* Compact Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border">
        <div className="flex items-center space-x-3">
          {tool.mcp_info.logo_url && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={resolveLogoSrc(tool.mcp_info.logo_url)}
              alt={`${tool.mcp_info.server_name} logo`}
              className="w-6 h-6 object-contain"
            />
          )}
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h2 className="text-lg font-semibold text-foreground">Test Tool:</h2>
              <div
                className="group inline-flex items-center space-x-1 bg-muted hover:bg-accent px-3 py-1 rounded-md cursor-pointer transition-colors border border-border"
                onClick={handleCopyToolName}
                title="Click to copy tool name"
              >
                <span className="font-mono text-foreground font-medium text-sm">{tool.name}</span>
                <svg
                  className="w-3 h-3 text-muted-foreground group-hover:text-foreground transition-colors"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
              </div>
            </div>
            <p className="text-xs text-muted-foreground">{tool.description}</p>
            <p className="text-xs text-muted-foreground">Provider: {tool.mcp_info.server_name}</p>
          </div>
        </div>
        <Button
          onClick={onClose}
          variant="ghost"
          size="icon-sm"
          aria-label="Close"
          className="text-muted-foreground hover:text-foreground"
        >
          <X className="size-4" />
        </Button>
      </div>

      {/* Two Column Layout - Always Side by Side */}
      <div className="grid grid-cols-2 gap-4 h-full">
        {/* Left Column - Input Parameters */}
        <div className="bg-card border border-border rounded-lg">
          <div className="border-b border-border px-4 py-2">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-foreground">Input Parameters</h3>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger
                    render={<CircleHelp className="size-4 cursor-help text-muted-foreground hover:text-foreground" />}
                  />
                  <TooltipContent>Configure the input parameters for this tool call</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>

          <div className="p-4">
            <ToolArgumentsForm
              key={formKey}
              fields={argumentFields}
              singleInputFallback={typeof tool.inputSchema === "string"}
              isLoading={isLoading}
              hasRun={Boolean(result || error)}
              onRun={runToolCall}
            />
          </div>
        </div>

        {/* Right Column - Tool Result */}
        <div className="bg-card border border-border rounded-lg">
          <div className="border-b border-border px-4 py-2">
            <h3 className="text-sm font-semibold text-foreground">Tool Result</h3>
          </div>

          <div className="p-4">
            {!result && !error && !isLoading ? (
              /* Empty State */
              <div className="flex flex-col justify-center items-center h-48 text-muted-foreground">
                <div className="text-center max-w-sm">
                  <div className="mb-3">
                    <svg
                      className="mx-auto h-12 w-12 text-muted-foreground"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1}
                        d="M13 10V3L4 14h7v7l9-11h-7z"
                      />
                    </svg>
                  </div>
                  <h4 className="text-sm font-medium text-foreground mb-1">Ready to Call Tool</h4>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Configure the input parameters and click &quot;Call Tool&quot; to see the results here.
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {/* Result Control Bar */}
                {result && !isLoading && !error && (
                  <div className="p-2 bg-success/10 border border-success/20 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <svg className="h-4 w-4 text-success" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        <h4 className="text-xs font-medium text-success">Tool executed successfully</h4>
                        {duration !== null && (
                          <span className="text-xs text-success ml-1">• {(duration / 1000).toFixed(2)}s</span>
                        )}
                      </div>

                      <div className="flex items-center space-x-1">
                        <div className="flex bg-card rounded-sm border border-success/30 p-0.5">
                          <button
                            onClick={() => setViewMode("formatted")}
                            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                              viewMode === "formatted"
                                ? "bg-success/15 text-success"
                                : "text-success hover:text-success/80"
                            }`}
                          >
                            Formatted
                          </button>
                          <button
                            onClick={() => setViewMode("json")}
                            className={`px-2 py-1 text-xs font-medium rounded transition-colors ${
                              viewMode === "json" ? "bg-success/15 text-success" : "text-success hover:text-success/80"
                            }`}
                          >
                            JSON
                          </button>
                        </div>

                        <button
                          onClick={handleCopyResult}
                          className="p-1 hover:bg-success/15 rounded-sm text-success"
                          title="Copy response"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="14"
                            height="14"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                <div className="max-h-96 overflow-y-auto">
                  {isLoading && (
                    <div className="flex flex-col justify-center items-center h-48 text-muted-foreground">
                      <div className="relative">
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-border"></div>
                        <div className="animate-spin rounded-full h-8 w-8 border-2 border-info border-t-transparent absolute top-0"></div>
                      </div>
                      <p className="text-sm font-medium mt-3">Calling tool...</p>
                      <p className="text-xs text-muted-foreground mt-1">Please wait while we process your request</p>
                    </div>
                  )}

                  {error && (
                    <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-3">
                      <div className="flex items-start space-x-2">
                        <div className="shrink-0">
                          <svg
                            className="h-4 w-4 text-destructive"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                          </svg>
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center space-x-2 mb-1">
                            <h4 className="text-xs font-medium text-destructive">Tool Call Failed</h4>
                            {duration !== null && (
                              <span className="text-xs text-destructive">• {(duration / 1000).toFixed(2)}s</span>
                            )}
                          </div>
                          <div className="bg-card border border-destructive/20 rounded-sm p-2 max-h-48 overflow-y-auto">
                            <pre className="text-xs whitespace-pre-wrap text-destructive font-mono">
                              {(() => {
                                return error.message;
                              })()}
                            </pre>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {result && !isLoading && !error && (
                    <div className="space-y-3">
                      {viewMode === "formatted" ? (
                        // Formatted View
                        result.map((content: any, idx: number) => (
                          <div key={idx} className="border border-border rounded-lg overflow-hidden">
                            {content.type === "text" && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-foreground uppercase tracking-wide">
                                    Text Response
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="bg-card rounded-sm border border-border max-h-64 overflow-y-auto">
                                    <div className="p-3 space-y-2">
                                      {content.text
                                        .split("\n\n")
                                        .map((section: string, sectionIndex: number) => {
                                          if (section.trim() === "") return null;

                                          // Handle headers (## or ###)
                                          if (section.startsWith("##")) {
                                            const headerText = section.replace(/^#+\s/, "");
                                            return (
                                              <div key={sectionIndex} className="border-b border-border pb-1 mb-2">
                                                <h3 className="text-sm font-semibold text-foreground">{headerText}</h3>
                                              </div>
                                            );
                                          }

                                          // Handle URL-containing sections
                                          const urlRegex = /(https?:\/\/[^\s\)]+)/g;
                                          if (urlRegex.test(section)) {
                                            const parts = section.split(urlRegex);
                                            return (
                                              <div
                                                key={sectionIndex}
                                                className="bg-info/10 border border-info/20 rounded-sm p-2"
                                              >
                                                <div className="text-xs text-foreground leading-relaxed whitespace-pre-wrap">
                                                  {parts.map((part, partIndex) => {
                                                    if (urlRegex.test(part)) {
                                                      return (
                                                        <a
                                                          key={partIndex}
                                                          href={part}
                                                          target="_blank"
                                                          rel="noopener noreferrer"
                                                          className="text-info hover:text-info/80 underline break-all"
                                                        >
                                                          {part}
                                                        </a>
                                                      );
                                                    }
                                                    return part;
                                                  })}
                                                </div>
                                              </div>
                                            );
                                          }

                                          // Handle score information
                                          if (section.includes("Score:")) {
                                            return (
                                              <div
                                                key={sectionIndex}
                                                className="bg-success/10 border-l-4 border-success p-2 rounded-r"
                                              >
                                                <p className="text-xs text-success font-medium whitespace-pre-wrap">
                                                  {section}
                                                </p>
                                              </div>
                                            );
                                          }

                                          // Regular content sections
                                          return (
                                            <div
                                              key={sectionIndex}
                                              className="bg-muted rounded-sm p-2 border border-border"
                                            >
                                              <div className="text-xs text-foreground leading-relaxed whitespace-pre-wrap font-mono">
                                                {section}
                                              </div>
                                            </div>
                                          );
                                        })
                                        .filter(Boolean)}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "image" && content.url && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-foreground uppercase tracking-wide">
                                    Image Response
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="bg-muted rounded-sm p-3 border border-border">
                                    {/* eslint-disable-next-line @next/next/no-img-element */}
                                    <img
                                      src={content.url}
                                      alt="Tool result"
                                      className="max-w-full h-auto rounded-sm shadow-xs"
                                    />
                                  </div>
                                </div>
                              </div>
                            )}

                            {content.type === "embedded_resource" && (
                              <div>
                                <div className="bg-muted px-3 py-1 border-b border-border">
                                  <span className="text-xs font-medium text-foreground uppercase tracking-wide">
                                    Embedded Resource
                                  </span>
                                </div>
                                <div className="p-3">
                                  <div className="flex items-center space-x-2 p-3 bg-info/10 border border-info/20 rounded-sm">
                                    <div className="shrink-0">
                                      <svg
                                        className="h-5 w-5 text-info"
                                        fill="none"
                                        viewBox="0 0 24 24"
                                        stroke="currentColor"
                                      >
                                        <path
                                          strokeLinecap="round"
                                          strokeLinejoin="round"
                                          strokeWidth={2}
                                          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                                        />
                                      </svg>
                                    </div>
                                    <div className="flex-1">
                                      <p className="text-xs font-medium text-info">
                                        Resource Type: {content.resource_type}
                                      </p>
                                      {content.url && (
                                        <a
                                          href={content.url}
                                          target="_blank"
                                          rel="noopener noreferrer"
                                          className="inline-flex items-center text-xs text-info hover:underline mt-1"
                                        >
                                          View Resource
                                          <svg className="ml-1 h-3 w-3" fill="currentColor" viewBox="0 0 20 20">
                                            <path d="M11 3a1 1 0 100 2h2.586l-6.293 6.293a1 1 0 101.414 1.414L15 6.414V9a1 1 0 102 0V4a1 1 0 00-1-1h-5z" />
                                            <path d="M5 5a2 2 0 00-2 2v8a2 2 0 002 2h8a2 2 0 002-2v-3a1 1 0 10-2 0v3H5V7h3a1 1 0 000-2H5z" />
                                          </svg>
                                        </a>
                                      )}
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        ))
                      ) : (
                        // JSON View
                        <div className="bg-card rounded-sm border border-border">
                          <div className="p-3 overflow-auto max-h-80 bg-muted">
                            <pre className="text-xs font-mono whitespace-pre-wrap break-all text-foreground">
                              {JSON.stringify(result, null, 2)}
                            </pre>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
