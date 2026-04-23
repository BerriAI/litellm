import React from "react";
import { Button } from "@/components/ui/button";
import {
  AlertTriangle,
  CheckCircle2,
  Copy,
  Info,
  Loader2,
} from "lucide-react";
import { testConnectionRequest } from "../networking";
import { prepareModelAddRequest } from "./handle_add_model_submit";
import NotificationsManager from "../molecules/notifications_manager";

interface ModelConnectionTestProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  modelName = "this model",
  onTestComplete,
}) => {
  const [error, setError] = React.useState<Error | string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [rawResponse, setRawResponse] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState<boolean>(true);
  const [isSuccess, setIsSuccess] = React.useState<boolean>(false);
  const [showDetails, setShowDetails] = React.useState<boolean>(false);

  React.useEffect(() => {
    const testModelConnection = async () => {
      setIsLoading(true);
      setShowDetails(false);
      setError(null);
      setRawResponse(null);
      setIsSuccess(false);

      await new Promise((resolve) => setTimeout(resolve, 100));

      try {
        const result = await prepareModelAddRequest(
          formValues,
          accessToken,
          null,
        );

        if (!result) {
          setError(
            "Failed to prepare model data. Please check your form inputs.",
          );
          setIsSuccess(false);
          setIsLoading(false);
          return;
        }

        const { litellmParamsObj, modelInfoObj } = result[0];

        const response = await testConnectionRequest(
          accessToken,
          litellmParamsObj,
          modelInfoObj,
          modelInfoObj?.mode,
        );
        if (response.status === "success") {
          NotificationsManager.success("Connection test successful!");
          setError(null);
          setIsSuccess(true);
        } else {
          const errorMessage =
            response.result?.error || response.message || "Unknown error";
          setError(errorMessage);
          setRawResponse(response.result?.raw_request_typed_dict);
          setIsSuccess(false);
        }
      } catch (err) {
        console.error("Test connection error:", err);
        setError(err instanceof Error ? err.message : String(err));
        setIsSuccess(false);
      } finally {
        setIsLoading(false);
        if (onTestComplete) onTestComplete();
      }
    };

    const timer = setTimeout(() => {
      testModelConnection();
    }, 200);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const getCleanErrorMessage = (errorMsg: string) => {
    if (!errorMsg) return "Unknown error";

    const mainError = errorMsg.split("stack trace:")[0].trim();
    const cleanedError = mainError.replace(/^litellm\.(.*?)Error: /, "");
    return cleanedError;
  };

  const errorMessage =
    typeof error === "string"
      ? getCleanErrorMessage(error)
      : error?.message
        ? getCleanErrorMessage(error.message)
        : "Unknown error";

  const formatCurlCommand = (
    apiBase: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  ${headerString ? `${headerString} \\\n  ` : ""}-H 'Content-Type: application/json' \\
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
    <div className="p-6 rounded-lg bg-background">
      {isLoading ? (
        <div className="text-center py-8 px-5">
          <div className="mb-4 flex justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
          <span className="text-base">Testing connection to {modelName}...</span>
        </div>
      ) : isSuccess ? (
        <div className="flex items-center justify-center py-8 px-5">
          {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
          <CheckCircle2 className="h-7 w-7 text-emerald-500" />
          <span
            data-testid="connection-success-msg"
            // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
            className="text-emerald-600 text-lg font-medium ml-2.5"
          >
            Connection to {modelName} successful!
          </span>
        </div>
      ) : (
        <div>
          <div className="flex items-center mb-5">
            <AlertTriangle className="h-6 w-6 text-destructive mr-3" />
            <span
              data-testid="connection-failure-msg"
              className="text-destructive text-lg font-medium"
            >
              Connection to {modelName} failed
            </span>
          </div>

          <div className="bg-destructive/5 border border-destructive/30 rounded-lg p-4 mb-5">
            <span className="block mb-2 font-semibold">Error: </span>
            <span className="text-destructive text-sm leading-relaxed">
              {errorMessage}
            </span>

            {error && (
              <div className="mt-3">
                <Button
                  type="button"
                  variant="link"
                  onClick={() => setShowDetails(!showDetails)}
                  className="p-0 h-auto"
                >
                  {showDetails ? "Hide Details" : "Show Details"}
                </Button>
              </div>
            )}
          </div>

          {showDetails && (
            <div className="mb-5">
              <span className="block mb-2 text-[15px] font-semibold">
                Troubleshooting Details
              </span>
              <pre className="bg-muted p-4 rounded-lg text-xs max-h-[200px] overflow-auto border border-border leading-relaxed">
                {typeof error === "string"
                  ? error
                  : JSON.stringify(error, null, 2)}
              </pre>
            </div>
          )}

          <div>
            <span className="block mb-2 text-[15px] font-semibold">
              API Request
            </span>
            <pre className="bg-muted p-4 rounded-lg text-xs max-h-[250px] overflow-auto border border-border leading-relaxed">
              {curlCommand || "No request data available"}
            </pre>
            <Button
              type="button"
              variant="outline"
              className="mt-2"
              onClick={() => {
                navigator.clipboard.writeText(curlCommand || "");
                NotificationsManager.success("Copied to clipboard");
              }}
            >
              <Copy className="h-4 w-4" />
              Copy to Clipboard
            </Button>
          </div>
        </div>
      )}
      <hr className="border-border my-6" />
      <div className="flex justify-between items-center">
        <Button asChild variant="link" className="p-0 h-auto">
          <a
            href="https://docs.litellm.ai/docs/providers"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Info className="h-4 w-4" />
            View Documentation
          </a>
        </Button>
      </div>
    </div>
  );
};

export default ModelConnectionTest;
