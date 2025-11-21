import { PromptType } from "./types";

export const extractVariables = (prompt: PromptType): string[] => {
  const variableSet = new Set<string>();
  const variableRegex = /\{\{(\w+)\}\}/g;

  prompt.messages.forEach((message) => {
    let match;
    while ((match = variableRegex.exec(message.content)) !== null) {
      variableSet.add(match[1]);
    }
  });

  if (prompt.developerMessage) {
    let match;
    while ((match = variableRegex.exec(prompt.developerMessage)) !== null) {
      variableSet.add(match[1]);
    }
  }

  return Array.from(variableSet);
};

export const convertToDotPrompt = (prompt: PromptType): string => {
  const variables = extractVariables(prompt);
  let result = `---\nmodel: ${prompt.model}\n`;

  // Add temperature if set
  if (prompt.config.temperature !== undefined) {
    result += `temperature: ${prompt.config.temperature}\n`;
  }

  // Add max_tokens if set
  if (prompt.config.max_tokens !== undefined) {
    result += `max_tokens: ${prompt.config.max_tokens}\n`;
  }

  // Add top_p if set
  if (prompt.config.top_p !== undefined) {
    result += `top_p: ${prompt.config.top_p}\n`;
  }

  // Add input schema
  result += `input:\n  schema:\n`;
  variables.forEach((variable) => {
    result += `    ${variable}: string\n`;
  });

  // Add output format
  result += `output:\n  format: text\n`;

  // Add tools if present
  if (prompt.tools && prompt.tools.length > 0) {
    result += `tools:\n`;
    prompt.tools.forEach((tool) => {
      const toolObj = JSON.parse(tool.json);
      result += `  - ${JSON.stringify(toolObj)}\n`;
    });
  }

  result += `---\n\n`;

  // Add developer message if present
  if (prompt.developerMessage && prompt.developerMessage.trim() !== "") {
    result += `Developer: ${prompt.developerMessage.trim()}\n\n`;
  }

  // Add messages with role prefixes
  prompt.messages.forEach((message) => {
    const role = message.role.charAt(0).toUpperCase() + message.role.slice(1);
    result += `${role}: ${message.content}\n\n`;
  });

  return result.trim();
};

