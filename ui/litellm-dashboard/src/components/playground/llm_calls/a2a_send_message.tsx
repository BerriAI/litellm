// a2a_send_message.tsx
// A2A Protocol (JSON-RPC 2.0) implementation for sending messages to agents

import { v4 as uuidv4 } from "uuid";
import { getProxyBaseUrl, getGlobalLitellmHeaderName } from "../../networking";
import { A2ATaskMetadata } from "../chat_ui/types";

interface A2AMessagePart {
  kind: "text";
  text: string;
}

interface A2AMessage {
  kind: "message";
  messageId: string;
  role: "user" | "agent";
  parts: A2AMessagePart[];
}

interface A2AJsonRpcRequest {
  jsonrpc: "2.0";
  id: string;
  method: string;
  params: {
    message: A2AMessage;
    metadata?: { guardrails?: string[] };
  };
}

interface A2AJsonRpcResponse {
  jsonrpc: "2.0";
  id: string;
  result?: {
    kind?: string;
    parts?: A2AMessagePart[];
    id?: string;
    contextId?: string;
    status?: {
      state?: string;
      timestamp?: string;
      message?: {
        parts?: A2AMessagePart[];
      };
    };
    metadata?: Record<string, any>;
    artifacts?: Array<{
      artifactId?: string;
      name?: string;
      parts?: A2AMessagePart[];
    }>;
    [key: string]: any;
  };
  error?: {
    code: number;
    message: string;
  };
}

/**
 * Extracts A2A task metadata from the response result.
 */
const extractA2AMetadata = (result: A2AJsonRpcResponse["result"]): A2ATaskMetadata | undefined => {
  if (!result) return undefined;

  const metadata: A2ATaskMetadata = {};

  // Extract task ID
  if (result.id) {
    metadata.taskId = result.id;
  }

  // Extract context/session ID
  if (result.contextId) {
    metadata.contextId = result.contextId;
  }

  // Extract status
  if (result.status) {
    metadata.status = {
      state: result.status.state,
      timestamp: result.status.timestamp,
    };

    // Extract status message text if present
    if (result.status.message?.parts) {
      const statusText = result.status.message.parts
        .filter((p: any) => p.kind === "text" && p.text)
        .map((p: any) => p.text)
        .join(" ");
      if (statusText) {
        metadata.status.message = statusText;
      }
    }
  }

  // Extract custom metadata
  if (result.metadata && typeof result.metadata === "object") {
    metadata.metadata = result.metadata;
  }

  return Object.keys(metadata).length > 0 ? metadata : undefined;
};

/**
 * Sends a message to an A2A agent using the JSON-RPC 2.0 protocol.
 * Uses the non-streaming message/send method.
 */
export const makeA2ASendMessageRequest = async (
  agentId: string,
  message: string,
  onTextUpdate: (chunk: string, model?: string) => void,
  accessToken: string,
  signal?: AbortSignal,
  onTimingData?: (timeToFirstToken: number) => void,
  onTotalLatency?: (totalLatency: number) => void,
  onA2AMetadata?: (metadata: A2ATaskMetadata) => void,
  customBaseUrl?: string,
  guardrails?: string[],
): Promise<void> => {
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/a2a/${agentId}/message/send`
    : `/a2a/${agentId}/message/send`;

  const requestId = uuidv4();
  const messageId = uuidv4().replace(/-/g, "");

  const jsonRpcRequest: A2AJsonRpcRequest = {
    jsonrpc: "2.0",
    id: requestId,
    method: "message/send",
    params: {
      message: {
        kind: "message",
        messageId: messageId,
        role: "user",
        parts: [{ kind: "text", text: message }],
      },
    },
  };

  if (guardrails && guardrails.length > 0) {
    jsonRpcRequest.params.metadata = { guardrails };
  }

  const startTime = performance.now();

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(jsonRpcRequest),
      signal,
    });

    const timeToFirstToken = performance.now() - startTime;
    if (onTimingData) {
      onTimingData(timeToFirstToken);
    }

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || errorData.detail || `HTTP ${response.status}`);
    }

    const jsonRpcResponse: A2AJsonRpcResponse = await response.json();

    const totalLatency = performance.now() - startTime;
    if (onTotalLatency) {
      onTotalLatency(totalLatency);
    }

    if (jsonRpcResponse.error) {
      throw new Error(jsonRpcResponse.error.message);
    }

    // Extract text and metadata from response
    const result = jsonRpcResponse.result;
    if (result) {
      let responseText = "";

      // Extract and send A2A metadata
      const a2aMetadata = extractA2AMetadata(result);
      if (a2aMetadata && onA2AMetadata) {
        onA2AMetadata(a2aMetadata);
      }

      // A2A Task response format with artifacts array
      // Extract text from artifacts[*].parts[*] where kind === "text"
      if (result.artifacts && Array.isArray(result.artifacts)) {
        for (const artifact of result.artifacts) {
          if (artifact.parts && Array.isArray(artifact.parts)) {
            for (const part of artifact.parts) {
              if (part.kind === "text" && part.text) {
                responseText += part.text;
              }
            }
          }
        }
      }
      // Fallback: direct parts array (simpler response format)
      else if (result.parts && Array.isArray(result.parts)) {
        for (const part of result.parts) {
          if (part.kind === "text" && part.text) {
            responseText += part.text;
          }
        }
      }
      // Fallback: status.message.parts format
      else if (result.status?.message?.parts) {
        for (const part of result.status.message.parts) {
          if (part.kind === "text" && part.text) {
            responseText += part.text;
          }
        }
      }

      if (responseText) {
        onTextUpdate(responseText, `a2a_agent/${agentId}`);
      } else {
        // Fallback: show raw result if we couldn't parse it
        console.warn("Could not extract text from A2A response, showing raw JSON:", result);
        onTextUpdate(JSON.stringify(result, null, 2), `a2a_agent/${agentId}`);
      }
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("A2A request was cancelled");
      return;
    }
    console.error("A2A send message error:", error);
    throw error;
  }
};

/**
 * Sends a streaming message to an A2A agent using the JSON-RPC 2.0 protocol.
 * Uses the message/stream method with NDJSON responses.
 */
export const makeA2AStreamMessageRequest = async (
  agentId: string,
  message: string,
  onTextUpdate: (chunk: string, model?: string) => void,
  accessToken: string,
  signal?: AbortSignal,
  onTimingData?: (timeToFirstToken: number) => void,
  onTotalLatency?: (totalLatency: number) => void,
  onA2AMetadata?: (metadata: A2ATaskMetadata) => void,
  customBaseUrl?: string,
): Promise<void> => {
  const proxyBaseUrl = customBaseUrl || getProxyBaseUrl();
  const url = proxyBaseUrl
    ? `${proxyBaseUrl}/a2a/${agentId}`
    : `/a2a/${agentId}`;

  const requestId = uuidv4();
  const messageId = uuidv4().replace(/-/g, "");

  const jsonRpcRequest: A2AJsonRpcRequest = {
    jsonrpc: "2.0",
    id: requestId,
    method: "message/stream",
    params: {
      message: {
        kind: "message",
        messageId: messageId,
        role: "user",
        parts: [{ kind: "text", text: message }],
      },
    },
  };

  const startTime = performance.now();
  let firstChunkReceived = false;
  let latestMetadata: A2ATaskMetadata | undefined;
  let accumulatedText = "";

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(jsonRpcRequest),
      signal,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error?.message || errorData.detail || `HTTP ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";
    let done = false;

    while (!done) {
      const readResult = await reader.read();
      done = readResult.done;
      const value = readResult.value;
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const chunk = JSON.parse(line);

          if (!firstChunkReceived) {
            firstChunkReceived = true;
            const timeToFirstToken = performance.now() - startTime;
            if (onTimingData) {
              onTimingData(timeToFirstToken);
            }
          }

          // Handle streaming chunks - extract text from various A2A formats
          const result = chunk.result;
          if (result) {
            // Extract metadata from each chunk (keep latest)
            const chunkMetadata = extractA2AMetadata(result);
            if (chunkMetadata) {
              latestMetadata = { ...latestMetadata, ...chunkMetadata };
            }

            const chunkKind = result.kind;

            // Handle artifact-update chunks (streaming response content)
            // Note: streaming uses "artifact" (singular), not "artifacts" (plural)
            if (chunkKind === "artifact-update" && result.artifact) {
              const artifact = result.artifact;
              if (artifact.parts && Array.isArray(artifact.parts)) {
                for (const part of artifact.parts) {
                  if (part.kind === "text" && part.text) {
                    accumulatedText += part.text;
                    onTextUpdate(accumulatedText, `a2a_agent/${agentId}`);
                  }
                }
              }
            }
            // Handle non-streaming Task response format with artifacts array (plural)
            else if (result.artifacts && Array.isArray(result.artifacts)) {
              for (const artifact of result.artifacts) {
                if (artifact.parts && Array.isArray(artifact.parts)) {
                  for (const part of artifact.parts) {
                    if (part.kind === "text" && part.text) {
                      accumulatedText += part.text;
                      onTextUpdate(accumulatedText, `a2a_agent/${agentId}`);
                    }
                  }
                }
              }
            }
            // Handle status-update chunks (progress messages like "Processing request...")
            // These are metadata/status updates, not actual response content
            // We skip showing them in the chat UI - they're captured in metadata instead
            else if (chunkKind === "status-update") {
              // Status updates are handled via metadata extraction, not shown as text
              // This prevents "Processing request..." from appearing in the response
            }
            // Direct parts array (fallback)
            else if (result.parts && Array.isArray(result.parts)) {
              for (const part of result.parts) {
                if (part.kind === "text" && part.text) {
                  accumulatedText += part.text;
                  onTextUpdate(accumulatedText, `a2a_agent/${agentId}`);
                }
              }
            }
          }

          // Handle JSON-RPC error response
          if (chunk.error) {
            const errorMessage = chunk.error.message || "Unknown A2A error";
            throw new Error(errorMessage);
          }
        } catch (parseError) {
          // Re-throw if it's an actual error we threw (not a parse error)
          if (parseError instanceof Error && parseError.message && !parseError.message.includes("JSON")) {
            throw parseError;
          }
          // Only warn if it's not a JSON parse error on an empty/partial line
          if (line.trim().length > 0) {
            console.warn("Failed to parse A2A streaming chunk:", line, parseError);
          }
        }
      }
    }

    const totalLatency = performance.now() - startTime;
    if (onTotalLatency) {
      onTotalLatency(totalLatency);
    }

    // Send final metadata after streaming completes
    if (latestMetadata && onA2AMetadata) {
      onA2AMetadata(latestMetadata);
    }
  } catch (error) {
    if (signal?.aborted) {
      console.log("A2A streaming request was cancelled");
      return;
    }
    console.error("A2A stream message error:", error);
    throw error;
  }
};
