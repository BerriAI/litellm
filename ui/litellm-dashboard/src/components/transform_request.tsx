import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Copy } from "lucide-react";
import { transformRequestCall } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
interface TransformRequestPanelProps {
  accessToken: string | null;
}

interface TransformResponse {
  raw_request_api_base: string;
  raw_request_body: Record<string, any>;
  raw_request_headers: Record<string, string>;
}

const TransformRequestPanel: React.FC<TransformRequestPanelProps> = ({ accessToken }) => {
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
        NotificationsManager.fromBackend("Invalid JSON in request body");
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
        NotificationsManager.fromBackend("No access token found");
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
        NotificationsManager.success("Request transformed successfully");
      } else {
        // Handle the case where the API returns a different format
        // Try to extract the parts from a string response if needed
        const rawText = typeof data === "string" ? data : JSON.stringify(data);
        setTransformedResponse(rawText);
        NotificationsManager.info("Transformed request received in unexpected format");
      }
    } catch (err) {
      console.error("Error transforming request:", err);
      NotificationsManager.fromBackend("Failed to transform request");
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
    <div className="w-full m-2 overflow-hidden">
      <h1 className="text-2xl font-semibold">Playground</h1>
      <p className="text-sm text-muted-foreground">
        See how LiteLLM transforms your request for the specified provider.
      </p>
      <div className="flex gap-4 w-full min-w-0 overflow-hidden mt-4">
        {/* Original Request Panel */}
        <div className="flex-1 basis-1/2 flex flex-col border border-border rounded-lg p-6 overflow-hidden max-h-[600px] min-w-0">
          <div className="mb-6">
            <h2 className="text-2xl font-bold mb-1">Original Request</h2>
            <p className="text-muted-foreground m-0">
              The request you would send to LiteLLM /chat/completions endpoint.
            </p>
          </div>

          <textarea
            className="flex-1 w-full min-h-[240px] p-4 border border-border rounded-md font-mono text-sm resize-none mb-6 overflow-auto"
            value={originalRequestJSON}
            onChange={(e) => setOriginalRequestJSON(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Press Cmd/Ctrl + Enter to transform"
          />

          <div className="flex justify-end mt-auto">
            <Button
              onClick={handleTransform}
              disabled={isLoading}
              className="bg-black text-white hover:bg-black/90"
            >
              <span>{isLoading ? "Transforming..." : "Transform"}</span>
              <span>→</span>
            </Button>
          </div>
        </div>

        {/* Transformed Request Panel */}
        <div className="flex-1 basis-1/2 flex flex-col border border-border rounded-lg p-6 overflow-hidden max-h-[800px] min-w-0">
          <div className="mb-6">
            <h2 className="text-2xl font-bold mb-1">Transformed Request</h2>
            <p className="text-muted-foreground m-0">
              How LiteLLM transforms your request for the specified provider.
            </p>
            <br />
            <p className="text-muted-foreground m-0 text-xs">
              Note: Sensitive headers are not shown.
            </p>
          </div>

          <div className="relative bg-muted rounded-md flex-1 flex flex-col overflow-hidden">
            <pre className="p-4 font-mono text-sm m-0 overflow-auto flex-1">
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
              size="icon"
              className="absolute right-2 top-2 h-7 w-7"
              onClick={() => {
                navigator.clipboard.writeText(transformedResponse || "");
                NotificationsManager.success("Copied to clipboard");
              }}
              aria-label="Copy"
            >
              <Copy className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </div>
      <div className="mt-4 text-right w-full">
        <p className="text-sm text-muted-foreground">
          Found an error? File an issue{" "}
          <a
            href="https://github.com/BerriAI/litellm/issues"
            target="_blank"
            rel="noopener noreferrer"
            className="text-primary hover:text-primary/80 underline"
          >
            here
          </a>
          .
        </p>
      </div>
    </div>
  );
};

export default TransformRequestPanel;
