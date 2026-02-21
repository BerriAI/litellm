/**
 * Utility functions for parsing and processing tool data from log entries
 */

import { LogEntry } from "../columns";
import { ParsedTool, ToolDefinition, ToolCall } from "./types";

/**
 * Parse raw data that might be a string or object
 */
function parseData(input: any): any {
  if (typeof input === "string") {
    try {
      return JSON.parse(input);
    } catch {
      return input;
    }
  }
  return input;
}

/**
 * Extract tools array from request data
 */
function extractToolsFromRequest(log: LogEntry): ToolDefinition[] {
  // Check proxy_server_request first (most complete), then messages
  const requestData = parseData(log.proxy_server_request || log.messages);
  
  if (!requestData) return [];
  
  // Handle array format (messages array)
  if (Array.isArray(requestData)) {
    // Tools are not typically in messages array, return empty
    return [];
  }
  
  // Handle object format (request body)
  if (typeof requestData === "object" && requestData.tools) {
    return Array.isArray(requestData.tools) ? requestData.tools : [];
  }
  
  return [];
}

/**
 * Extract tool calls from response data
 */
function extractToolCallsFromResponse(log: LogEntry): ToolCall[] {
  const responseData = parseData(log.response);
  
  if (!responseData || typeof responseData !== "object") return [];
  
  // OpenAI format: response.choices[0].message.tool_calls
  const choices = responseData.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const firstChoice = choices[0];
    const message = firstChoice.message;
    if (message && Array.isArray(message.tool_calls)) {
      return message.tool_calls;
    }
  }
  
  return [];
}

/**
 * Parse safe JSON with fallback
 */
function parseSafeJson(jsonString: string): Record<string, any> {
  try {
    return JSON.parse(jsonString);
  } catch {
    return {};
  }
}

/**
 * Main function to parse tools from a log entry
 * Returns an array of tools with their definition and call status
 */
export function parseToolsFromLog(log: LogEntry): ParsedTool[] {
  // Get tools from request
  const requestTools = extractToolsFromRequest(log);
  
  if (requestTools.length === 0) {
    return [];
  }
  
  // Get tool calls from response
  const toolCalls = extractToolCallsFromResponse(log);
  const calledToolNames = new Set(
    toolCalls.map((tc: ToolCall) => tc.function?.name).filter(Boolean)
  );
  
  // Map tool calls by name for quick lookup
  const toolCallMap = new Map<string, any>();
  toolCalls.forEach((tc: ToolCall) => {
    const name = tc.function?.name;
    if (name) {
      toolCallMap.set(name, {
        id: tc.id,
        name: name,
        arguments: parseSafeJson(tc.function?.arguments || "{}"),
      });
    }
  });
  
  // Parse each tool definition
  return requestTools.map((tool: ToolDefinition, index: number) => {
    const func = tool.function || { name: `Tool ${index + 1}` };
    const name = func.name || `Tool ${index + 1}`;
    
    return {
      index: index + 1,
      name: name,
      description: func.description || "",
      parameters: func.parameters || {},
      called: calledToolNames.has(name),
      callData: toolCallMap.get(name),
    };
  });
}

/**
 * Check if a log entry has any tools
 */
export function hasTools(log: LogEntry): boolean {
  const requestTools = extractToolsFromRequest(log);
  return requestTools.length > 0;
}
