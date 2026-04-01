import React, { useState } from "react";
import { Button } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { Title } from "@tremor/react";
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
    <div className="w-full m-2" style={{ overflow: "hidden" }}>
      <Title>Playground</Title>
      <p className="text-sm text-gray-500">See how LiteLLM transforms your request for the specified provider.</p>
      <div
        style={{
          display: "flex",
          gap: "16px",
          width: "100%",
          minWidth: 0,
          overflow: "hidden",
        }}
        className="mt-4"
      >
        {/* Original Request Panel */}
        <div
          style={{
            flex: "1 1 50%",
            display: "flex",
            flexDirection: "column",
            border: "1px solid #e8e8e8",
            borderRadius: "8px",
            padding: "24px",
            overflow: "hidden",
            maxHeight: "600px",
            minWidth: 0,
          }}
        >
          <div style={{ marginBottom: "24px" }}>
            <h2 style={{ fontSize: "24px", fontWeight: "bold", margin: "0 0 4px 0" }}>Original Request</h2>
            <p style={{ color: "#666", margin: 0 }}>
              The request you would send to LiteLLM /chat/completions endpoint.
            </p>
          </div>

          <textarea
            style={{
              flex: "1 1 auto",
              width: "100%",
              minHeight: "240px",
              padding: "16px",
              border: "1px solid #e8e8e8",
              borderRadius: "6px",
              fontFamily: "monospace",
              fontSize: "14px",
              resize: "none",
              marginBottom: "24px",
              overflow: "auto",
            }}
            value={originalRequestJSON}
            onChange={(e) => setOriginalRequestJSON(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Press Cmd/Ctrl + Enter to transform"
          />

          <div
            style={{
              display: "flex",
              justifyContent: "flex-end",
              marginTop: "auto",
            }}
          >
            <Button
              type="primary"
              style={{
                backgroundColor: "#000",
                display: "flex",
                alignItems: "center",
                gap: "8px",
              }}
              onClick={handleTransform}
              loading={isLoading}
            >
              <span>Transform</span>
              <span>→</span>
            </Button>
          </div>
        </div>

        {/* Transformed Request Panel */}
        <div
          style={{
            flex: "1 1 50%",
            display: "flex",
            flexDirection: "column",
            border: "1px solid #e8e8e8",
            borderRadius: "8px",
            padding: "24px",
            overflow: "hidden",
            maxHeight: "800px",
            minWidth: 0,
          }}
        >
          <div style={{ marginBottom: "24px" }}>
            <h2 style={{ fontSize: "24px", fontWeight: "bold", margin: "0 0 4px 0" }}>Transformed Request</h2>
            <p style={{ color: "#666", margin: 0 }}>How LiteLLM transforms your request for the specified provider.</p>
            <br />
            <p style={{ color: "#666", margin: 0 }} className="text-xs">
              Note: Sensitive headers are not shown.
            </p>
          </div>

          <div
            style={{
              position: "relative",
              backgroundColor: "#f5f5f5",
              borderRadius: "6px",
              flex: "1 1 auto",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <pre
              style={{
                padding: "16px",
                fontFamily: "monospace",
                fontSize: "14px",
                margin: 0,
                overflow: "auto",
                flex: "1 1 auto",
              }}
            >
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
              type="text"
              icon={<CopyOutlined />}
              style={{
                position: "absolute",
                right: "8px",
                top: "8px",
              }}
              size="small"
              onClick={() => {
                navigator.clipboard.writeText(transformedResponse || "");
                NotificationsManager.success("Copied to clipboard");
              }}
            />
          </div>
        </div>
      </div>
      <div className="mt-4 text-right w-full">
        <p className="text-sm text-gray-500">
          Found an error? File an issue{" "}
          <a href="https://github.com/BerriAI/litellm/issues" target="_blank" rel="noopener noreferrer">
            here
          </a>
          .
        </p>
      </div>
    </div>
  );
};

export default TransformRequestPanel;
