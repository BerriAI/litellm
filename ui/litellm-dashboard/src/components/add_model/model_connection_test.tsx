import React from "react";
import { AlertTriangle, CircleCheck, Copy, ExternalLink, Info, LoaderCircle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";

import { toast } from "@/lib/toast";
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";

interface ModelConnectionTestProps {
  formValues: Record<string, any>;
  accessToken: string;
  testMode: string;
  modelName?: string;
  onClose?: () => void;
  onTestComplete?: () => void;
}

const ModelConnectionTest: React.FC<ModelConnectionTestProps> = ({
  formValues,
  accessToken,
  testMode: _testMode,
  modelName = "this model",
  onClose: _onClose,
  onTestComplete,
}) => {
  const [error, setError] = React.useState<Error | string | null>(null);
  const [rawResponse, setRawResponse] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const [isSuccess, setIsSuccess] = React.useState(false);
  const [showDetails, setShowDetails] = React.useState(false);

  const testModelConnection = async () => {
    setIsLoading(true);
    setShowDetails(false);
    setError(null);
    setRawResponse(null);
    setIsSuccess(false);

    await new Promise((resolve) => setTimeout(resolve, 100));

    try {
      const result = await prepareModelAddRequest(formValues, accessToken, null);

      if (!result) {
        setError("Failed to prepare model data. Please check your form inputs.");
        setIsSuccess(false);
        setIsLoading(false);
        return;
      }

      const { litellmParamsObj, modelInfoObj } = result[0];
      const response = await testConnectionRequest(accessToken, litellmParamsObj, modelInfoObj, modelInfoObj?.mode);

      if (response.status === "success") {
        toast.success("Connection test successful!");
        setError(null);
        setIsSuccess(true);
      } else {
        const errorMessage = response.result?.error || response.message || "Unknown error";
        setError(errorMessage);
        setRawResponse(response.result?.raw_request_typed_dict);
        setIsSuccess(false);
      }
    } catch (connectionError) {
      console.error("Test connection error:", connectionError);
      setError(connectionError instanceof Error ? connectionError.message : String(connectionError));
      setIsSuccess(false);
    } finally {
      setIsLoading(false);
      onTestComplete?.();
    }
  };

  React.useEffect(() => {
    const timer = setTimeout(() => {
      testModelConnection();
    }, 200);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the parent remounts the component to start a fresh connection test
  }, []);

  const getCleanErrorMessage = (errorMsg: string) => {
    if (!errorMsg) return "Unknown error";
    return errorMsg
      .split("stack trace:")[0]
      .trim()
      .replace(/^litellm\.(.*?)Error: /, "");
  };

  const errorMessage =
    typeof error === "string"
      ? getCleanErrorMessage(error)
      : error?.message
        ? getCleanErrorMessage(error.message)
        : "Unknown error";

  const formatCurlCommand = (
    apiBase: string,
    requestBody: Record<string, any>,
    requestHeaders: Record<string, string>,
  ) => {
    const formattedBody = JSON.stringify(requestBody, null, 2)
      .split("\n")
      .map((line) => `  ${line}`)
      .join("\n");

    const headerString = Object.entries(requestHeaders)
      .map(([key, value]) => `-H '${key}: ${value}'`)
      .join(" \\\n  ");

    return `curl -X POST \\
  ${apiBase} \\
  ${
    headerString
      ? `${headerString} \\
  `
      : ""
  }-H 'Content-Type: application/json' \\
  -d '{
${formattedBody}
  }'`;
  };

  const curlCommand = rawResponse
    ? formatCurlCommand(
        rawResponse.raw_request_api_base,
        rawResponse.raw_request_body,
        rawResponse.raw_request_headers || {},
      )
    : "";

  return (
    <div className="rounded-lg bg-background p-6">
      {isLoading ? (
        <div aria-busy="true" className="flex flex-col items-center justify-center gap-4 px-5 py-8 text-center">
          <LoaderCircle className="size-8 animate-spin text-primary" />
          <p className="text-base">Testing connection to {modelName}...</p>
        </div>
      ) : isSuccess ? (
        <div className="flex items-center justify-center gap-2.5 px-5 py-8">
          <CircleCheck className="size-6 text-primary" />
          <p data-testid="connection-success-msg" className="text-lg font-medium">
            Connection to {modelName} successful!
          </p>
        </div>
      ) : (
        <div>
          <div className="mb-5 flex items-center gap-3">
            <AlertTriangle className="size-6 text-destructive" />
            <p data-testid="connection-failure-msg" className="text-lg font-medium text-destructive">
              Connection to {modelName} failed
            </p>
          </div>

          <div className="mb-5 rounded-lg border border-destructive/30 bg-destructive/10 p-4 shadow-xs">
            <p className="mb-2 font-medium">Error:</p>
            <p className="text-sm leading-relaxed text-destructive">{errorMessage}</p>

            {error && (
              <Button
                type="button"
                variant="link"
                className="mt-3 h-auto px-0"
                onClick={() => setShowDetails((visible) => !visible)}
              >
                {showDetails ? "Hide Details" : "Show Details"}
              </Button>
            )}
          </div>

          {showDetails && (
            <div className="mb-5">
              <p className="mb-2 text-sm font-medium">Troubleshooting Details</p>
              <pre className="max-h-52 overflow-auto rounded-lg border bg-muted/50 p-4 text-xs leading-relaxed">
                {typeof error === "string" ? error : JSON.stringify(error, null, 2)}
              </pre>
            </div>
          )}

          <div>
            <p className="mb-2 text-sm font-medium">API Request</p>
            <pre className="max-h-64 overflow-auto rounded-lg border bg-muted/50 p-4 text-xs leading-relaxed">
              {curlCommand || "No request data available"}
            </pre>
            <Button
              type="button"
              variant="outline"
              className="mt-2"
              onClick={() => {
                navigator.clipboard.writeText(curlCommand || "");
                toast.success("Copied to clipboard");
              }}
            >
              <Copy data-icon="inline-start" />
              Copy to Clipboard
            </Button>
          </div>
        </div>
      )}

      <Separator className="my-6" />
      <Button
        variant="link"
        className="px-0"
        nativeButton={false}
        render={<a href="https://docs.litellm.ai/docs/providers" target="_blank" rel="noopener noreferrer" />}
      >
        <Info data-icon="inline-start" />
        View Documentation
        <ExternalLink data-icon="inline-end" />
      </Button>
    </div>
  );
};

export default ModelConnectionTest;
