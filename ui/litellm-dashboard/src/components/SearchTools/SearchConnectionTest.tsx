import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import React, { useEffect, useState } from "react";
import NotificationsManager from "../molecules/notifications_manager";
import { testSearchToolConnection } from "../networking";

interface SearchConnectionTestProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
        const result = await testSearchToolConnection(
          accessToken,
          litellmParams,
        );
        setTestResult(result);
        if (result.status === "success") {
          NotificationsManager.success("Connection test successful!");
        }
      } catch (error) {
        setTestResult({
          status: "error",
          message:
            error instanceof Error
              ? error.message
              : "Unknown error occurred",
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
      const titleMatch = finalError.match(/<title>(.*?)<\/title>/);
      if (titleMatch) {
        return titleMatch[1];
      }
      if (
        finalError.includes("401") ||
        finalError.includes("Authorization Required")
      ) {
        return "Authentication failed: Invalid API key or credentials";
      }
      return "Authentication error - please check your API key";
    }

    if (finalError.length > 200) {
      return finalError.substring(0, 200) + "...";
    }

    return finalError;
  };

  const errorMessage = testResult?.message
    ? getCleanErrorMessage(testResult.message)
    : "Unknown error";

  if (isLoading) {
    return (
      <div className="p-6 rounded-lg bg-background">
        <div className="text-center py-8 px-5">
          <div className="mb-4 flex justify-center">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
          <span className="text-base">
            Testing connection to{" "}
            {litellmParams.search_provider || "search provider"}...
          </span>
        </div>
      </div>
    );
  }

  if (!testResult) {
    return null;
  }

  return (
    <div className="p-6 rounded-lg bg-background">
      {testResult.status === "success" ? (
        <div className="flex items-center justify-center py-8 px-5">
          <CheckCircle2 className="h-7 w-7 text-emerald-500" />
          <div className="ml-3">
            <span className="text-emerald-600 text-lg font-medium block">
              Connection to {litellmParams.search_provider} successful!
            </span>
            {testResult.test_query && (
              <span className="text-sm text-muted-foreground mt-2 block">
                Test query:{" "}
                <code className="bg-muted px-1.5 py-0.5 rounded">
                  {testResult.test_query}
                </code>
              </span>
            )}
            {testResult.results_count !== undefined && (
              <span className="text-sm text-muted-foreground block">
                Results retrieved: {testResult.results_count}
              </span>
            )}
          </div>
        </div>
      ) : (
        <div>
          <div className="flex items-center mb-5">
            <AlertTriangle className="h-6 w-6 text-destructive mr-3" />
            <span className="text-destructive text-lg font-medium">
              Connection to{" "}
              {litellmParams.search_provider || "search provider"} failed
            </span>
          </div>

          <div className="bg-destructive/5 border border-destructive/30 rounded-lg p-4 mb-5">
            <span className="block mb-2 font-semibold">Error: </span>
            <span className="text-destructive text-sm leading-relaxed">
              {errorMessage}
            </span>

            {testResult.error_type && (
              <div className="mt-2">
                <span className="text-xs text-muted-foreground">
                  Error type:{" "}
                  <code className="bg-destructive/10 text-destructive px-1.5 py-0.5 rounded">
                    {testResult.error_type}
                  </code>
                </span>
              </div>
            )}

            {testResult.message && (
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
                Full Error Details
              </span>
              <pre className="bg-muted p-4 rounded-lg text-xs max-h-[200px] overflow-auto border border-border leading-relaxed whitespace-pre-wrap break-words">
                {testResult.message}
              </pre>
            </div>
          )}

          {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
          <div className="bg-amber-50 border border-amber-200 border-l-4 border-l-amber-500 rounded-lg p-4 dark:bg-amber-950/30 dark:border-amber-900">
            {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
            <span className="block mb-2 text-amber-700 font-semibold dark:text-amber-300">
              Troubleshooting tips:
            </span>
            {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
            <ul className="my-2 pl-5 list-disc text-amber-900 dark:text-amber-200 space-y-1.5">
              <li>Verify your API key is correct and active</li>
              <li>Check if the search provider service is operational</li>
              <li>
                Ensure you have sufficient credits/quota with the provider
              </li>
              <li>
                Review the provider&apos;s documentation for any additional
                requirements
              </li>
            </ul>
          </div>
        </div>
      )}
      <hr className="border-border my-6" />
      <div className="flex justify-between items-center">
        <Button asChild variant="link" className="p-0 h-auto">
          <a
            href="https://docs.litellm.ai/docs/search"
            target="_blank"
            rel="noopener noreferrer"
          >
            <Info className="h-4 w-4" />
            View Search Documentation
          </a>
        </Button>
      </div>
    </div>
  );
};

export default SearchConnectionTest;
