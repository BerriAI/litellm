/**
 * Utility functions for parsing and formatting messages for pretty view
 */

import { ParsedMessage, ParsedMessages, RoleStyle } from './prettyMessagesTypes';

/**
 * Role color styles for message cards - minimal, professional design
 * Color only used for labels and left border accent
 */
export const ROLE_STYLES: Record<string, RoleStyle> = {
  system: {
    background: 'transparent',
    borderColor: '#8c8c8c',
    label: 'SYSTEM',
    labelColor: '#8c8c8c',
  },
  user: {
    background: 'transparent',
    borderColor: '#1677ff',
    label: 'USER',
    labelColor: '#1677ff',
  },
  assistant: {
    background: 'transparent',
    borderColor: '#52c41a',
    label: 'ASSISTANT',
    labelColor: '#52c41a',
  },
  tool: {
    background: 'transparent',
    borderColor: '#fa8c16',
    label: 'TOOL RESULT',
    labelColor: '#fa8c16',
  },
};

/**
 * Parse request messages and response message from log data
 */
export const parseMessages = (request: any, response: any): ParsedMessages => {
  // Parse request messages
  const requestMessages: ParsedMessage[] = [];
  
  if (request?.messages && Array.isArray(request.messages)) {
    request.messages.forEach((msg: any) => {
      requestMessages.push({
        role: msg.role || 'user',
        content: parseMessageContent(msg.content),
        toolCallId: msg.tool_call_id,
      });
    });
  }

  // Parse response message
  let responseMessage: ParsedMessage | null = null;
  const responseMsg = response?.choices?.[0]?.message;
  
  if (responseMsg) {
    responseMessage = {
      role: responseMsg.role || 'assistant',
      content: responseMsg.content || '',
      toolCalls: parseToolCalls(responseMsg.tool_calls),
    };
  }

  return { requestMessages, responseMessage };
};

/**
 * Parse message content - handle strings and content arrays (for vision, etc.)
 */
const parseMessageContent = (content: any): string => {
  if (typeof content === 'string') {
    return content;
  }
  
  if (Array.isArray(content)) {
    // Handle content arrays (vision API format)
    return content
      .map((item) => {
        if (typeof item === 'string') return item;
        if (item.type === 'text') return item.text;
        if (item.type === 'image_url') return '[Image]';
        return JSON.stringify(item);
      })
      .join('\n');
  }
  
  // Fallback to JSON string for complex content
  return JSON.stringify(content);
};

/**
 * Parse tool calls from response message
 */
const parseToolCalls = (toolCalls: any[]): Array<{
  id: string;
  name: string;
  arguments: Record<string, any>;
}> | undefined => {
  if (!toolCalls || !Array.isArray(toolCalls)) return undefined;
  
  return toolCalls.map((tc) => ({
    id: tc.id || '',
    name: tc.function?.name || 'unknown',
    arguments: parseToolArguments(tc.function?.arguments),
  }));
};

/**
 * Parse tool arguments - handle both string and object formats
 */
const parseToolArguments = (args: any): Record<string, any> => {
  if (!args) return {};
  
  if (typeof args === 'string') {
    try {
      return JSON.parse(args);
    } catch {
      return { raw: args };
    }
  }
  
  return args;
};
