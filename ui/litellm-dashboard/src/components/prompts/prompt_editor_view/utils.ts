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
  let result = `---\nmodel: ${prompt.model}\n\ninput:\n  schema:\n`;

  variables.forEach((variable) => {
    result += `    ${variable}: string\n`;
  });

  result += `\noutput:\n  format: text\n---\n\n`;

  prompt.messages.forEach((message) => {
    result += `${message.role}: ${message.content}\n\n`;
  });

  return result;
};

