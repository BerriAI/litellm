import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState } from "react";
import { ArrowRight, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { transformRequestCall } from "@/components/networking";
import { toast } from "@/lib/toast";

interface TransformRequestPanelProps {
  accessToken: string | null;
}

const TransformRequestPanel: React.FC<TransformRequestPanelProps> = ({ accessToken }) => {
  const { t } = useLanguage();
  const [originalRequestJSON, setOriginalRequestJSON] = useState(`{
  "model": "openai/gpt-4o",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Explain quantum computing in simple terms"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 500,
  "stream": true
}`);

  const [transformedResponse, setTransformedResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Function to format curl command from API response parts
  const formatCurlCommand = (
    apiBase: string,
    requestBody: Record<string, any>,
    requestHeaders: Record<string, string>,
  ) => {
    // Format the request body as nicely indented JSON with 2 spaces
    const formattedBody = JSON.stringify(requestBody, null, 2)
      // Add additional indentation for the entire body
      .split("\n")
      .map((line) => `  ${line}`)
      .join("\n");

    // Build headers string with consistent indentation
    const headerString = Object.entries(requestHeaders)
      .map(([key, value]) => `-H '${key}: ${value}'`)
      .join(" \\\n  ");

    // Build the curl command with consistent indentation
    return `curl -X POST \\
  ${apiBase} \\
  ${headerString ? `${headerString} \\\n  ` : ""}-H 'Content-Type: application/json' \\
  -d '{
${formattedBody}
  }'`;
  };

  // Function to handle the transform request
  const handleTransform = async () => {
    setIsLoading(true);

    try {
      // Parse the JSON from the textarea
      let requestBody;
      try {
        requestBody = JSON.parse(originalRequestJSON);
      } catch (e) {
        toast.fromError(t("Invalid JSON in request body"));
        setIsLoading(false);
        return;
      }

      // Create the request payload
      const payload = {
        call_type: "completion",
        request_body: requestBody,
      };

      // Make the API call using fetch
      if (!accessToken) {
        toast.fromError(t("No access token found"));
        setIsLoading(false);
        return;
      }

      const data = await transformRequestCall(accessToken, payload);

      // Check if the response has the expected fields
      if (data.raw_request_api_base && data.raw_request_body) {
        // Format the curl command with the separate parts
        const formattedCurl = formatCurlCommand(
          data.raw_request_api_base,
          data.raw_request_body,
          data.raw_request_headers || {},
        );

        // Update state with the formatted curl command
        setTransformedResponse(formattedCurl);
        toast.success(t("Request transformed successfully"));
      } else {
        // Handle the case where the API returns a different format
        // Try to extract the parts from a string response if needed
        const rawText = typeof data === "string" ? data : JSON.stringify(data);
        setTransformedResponse(rawText);
        toast.info(t("Transformed request received in unexpected format"));
      }
    } catch (err) {
      console.error("Error transforming request:", err);
      toast.fromError(t("Failed to transform request"));
    } finally {
      setIsLoading(false);
    }
  };

  // Add this handler function near your other handlers
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault(); // Prevent default behavior
      handleTransform();
    }
  };

  return (
    <div className="p-2">
      <h1 className="text-lg font-medium text-foreground">{t("Playground")}</h1>
      <p className="text-sm text-muted-foreground">
        {t("See how LiteLLM transforms your request for the specified provider.")}
      </p>
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {/* Original Request Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl font-bold">{t("Original Request")}</CardTitle>
            <CardDescription>{t("The request you would send to LiteLLM /chat/completions endpoint.")}</CardDescription>
          </CardHeader>

          <CardContent>
            <Textarea
              className="h-72 resize-none p-4 font-mono text-sm field-sizing-fixed"
              value={originalRequestJSON}
              onChange={(e) => setOriginalRequestJSON(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t("Press Cmd/Ctrl + Enter to transform")}
            />
          </CardContent>

          <CardFooter className="justify-end">
            <Button onClick={handleTransform} disabled={isLoading}>
              <span>{t("Transform")}</span>
              {isLoading ? <UiLoadingSpinner className="size-4" /> : <ArrowRight />}
            </Button>
          </CardFooter>
        </Card>

        {/* Transformed Request Panel */}
        <Card>
          <CardHeader>
            <CardTitle className="text-2xl font-bold">{t("Transformed Request")}</CardTitle>
            <CardDescription>{t("How LiteLLM transforms your request for the specified provider.")}</CardDescription>
            <p className="mt-2 text-xs text-muted-foreground">{t("Note: Sensitive headers are not shown.")}</p>
          </CardHeader>

          <CardContent>
            <div className="relative rounded-md bg-muted">
              <pre className="h-72 overflow-auto p-4 font-mono text-sm">
                {transformedResponse ||
                  `curl -X POST \\
  https://api.openai.com/v1/chat/completions \\
  -H 'Authorization: Bearer sk-xxx' \\
  -H 'Content-Type: application/json' \\
  -d '{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    }
  ],
  "temperature": 0.7
  }'`}
              </pre>

              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={t("Copy to clipboard")}
                className="absolute top-2 right-2"
                onClick={() => {
                  navigator.clipboard.writeText(transformedResponse || "");
                  toast.success(t("Copied to clipboard"));
                }}
              >
                <Copy />
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
      <div className="mt-4 text-right">
        <p className="text-sm text-muted-foreground">
          {t("Found an error? File an issue")}{" "}
          <a
            className="underline underline-offset-4"
            href="https://github.com/BerriAI/litellm/issues"
            target="_blank"
            rel="noopener noreferrer"
          >
            {t("here")}
          </a>
          .
        </p>
      </div>
    </div>
  );
};

export default TransformRequestPanel;
