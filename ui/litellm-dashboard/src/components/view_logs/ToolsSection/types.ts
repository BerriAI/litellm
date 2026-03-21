/**
 * Type definitions for the Tools section
 */

export interface ToolDefinition {
  type: string;
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, any>;
  };
}

export interface ToolCall {
  id: string;
  type: string;
  function: {
    name: string;
    arguments: string;
  };
}

export interface ParsedTool {
  index: number;
  name: string;
  description: string;
  parameters: Record<string, any>;
  called: boolean;
  callData?: {
    id: string;
    name: string;
    arguments: Record<string, any>;
  };
}

export interface ParameterRow {
  key: string;
  name: string;
  type: string;
  description: string;
  required: boolean;
}
