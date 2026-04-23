import React from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  RefreshCcw,
  Wrench,
} from "lucide-react";

interface MCPConnectionStatusProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  formValues: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  tools: any[];
  isLoadingTools: boolean;
  toolsError: string | null;
  toolsErrorStackTrace: string | null;
  canFetchTools: boolean;
  fetchTools: () => Promise<void>;
}

const MCPConnectionStatus: React.FC<MCPConnectionStatusProps> = ({
  formValues,
  tools,
  isLoadingTools,
  toolsError,
  toolsErrorStackTrace,
  canFetchTools,
  fetchTools,
}) => {
  if (!canFetchTools && !formValues.url && !formValues.spec_path) {
    return null;
  }

  return (
    <Card className="p-4">
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-blue-600 dark:text-blue-400" />
          <h3 className="text-lg font-semibold">Connection Status</h3>
        </div>

        {!canFetchTools && (formValues.url || formValues.spec_path) && (
          <div className="text-center py-6 text-muted-foreground border border-dashed border-border rounded-lg">
            <Wrench className="h-6 w-6 mx-auto mb-2" />
            <div>Complete required fields to test connection</div>
            <div className="text-sm">
              Fill in URL, Transport, and Authentication to test MCP server
              connection
            </div>
          </div>
        )}

        {canFetchTools && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="text-foreground font-medium">
                  {isLoadingTools
                    ? "Testing connection to MCP server..."
                    : tools.length > 0
                      ? "Connection successful"
                      : toolsError
                        ? "Connection failed"
                        : "Ready to test connection"}
                </div>
                <div className="text-muted-foreground text-sm">
                  Server: {formValues.url || formValues.spec_path}
                </div>
              </div>

              {isLoadingTools && (
                <div className="flex items-center text-blue-600 dark:text-blue-400">
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  <span>Connecting...</span>
                </div>
              )}

              {!isLoadingTools && !toolsError && tools.length > 0 && (
                <div className="flex items-center text-emerald-600 dark:text-emerald-400">
                  <CheckCircle2 className="h-4 w-4 mr-1" />
                  <span className="font-medium">Connected</span>
                </div>
              )}

              {toolsError && (
                <div className="flex items-center text-destructive">
                  <AlertTriangle className="h-4 w-4 mr-1" />
                  <span className="font-medium">Failed</span>
                </div>
              )}
            </div>

            {isLoadingTools && (
              <div className="flex items-center justify-center py-6 gap-3">
                <Loader2 className="h-6 w-6 animate-spin text-primary" />
                <span>Testing connection and loading tools...</span>
              </div>
            )}

            {toolsError && (
              <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-md">
                <div className="flex justify-between items-start gap-2">
                  <div className="flex gap-2 flex-1">
                    <AlertTriangle className="h-4 w-4 mt-0.5 text-destructive shrink-0" />
                    <div className="flex-1 text-sm">
                      <div className="font-semibold text-destructive">
                        Connection Failed
                      </div>
                      <div>{toolsError}</div>
                      {toolsErrorStackTrace && (
                        <Accordion type="single" collapsible className="mt-3">
                          <AccordionItem value="stack-trace">
                            <AccordionTrigger>Stack Trace</AccordionTrigger>
                            <AccordionContent>
                              <pre className="whitespace-pre-wrap break-words text-xs font-mono m-0 p-2 bg-muted rounded max-h-[400px] overflow-auto">
                                {toolsErrorStackTrace}
                              </pre>
                            </AccordionContent>
                          </AccordionItem>
                        </Accordion>
                      )}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={fetchTools}
                    className="shrink-0"
                  >
                    <RefreshCcw className="h-3 w-3" />
                    Retry
                  </Button>
                </div>
              </div>
            )}

            {!isLoadingTools && tools.length === 0 && !toolsError && (
              <div className="text-center py-6 text-muted-foreground border border-dashed border-border rounded-lg">
                <CheckCircle2 className="h-6 w-6 mx-auto mb-2 text-emerald-500" />
                <div className="text-emerald-600 dark:text-emerald-400 font-medium">
                  Connection successful!
                </div>
                <div className="text-muted-foreground">
                  No tools found for this MCP server
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPConnectionStatus;
