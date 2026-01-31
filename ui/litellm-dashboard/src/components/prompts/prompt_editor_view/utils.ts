import { PromptType, Message, Tool } from "./types";

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

export const parseExistingPrompt = (apiResponse: any): PromptType => {
  // Extract dotprompt_content from litellm_params
  const dotpromptContent = apiResponse?.prompt_spec?.litellm_params?.dotprompt_content || "";
  
  if (!dotpromptContent) {
    throw new Error("No dotprompt_content found in API response");
  }

  // Split into frontmatter and content
  const parts = dotpromptContent.split("---");
  if (parts.length < 3) {
    throw new Error("Invalid dotprompt format");
  }

  // Parse YAML frontmatter (parts[1])
  const frontmatter = parts[1];
  const content = parts.slice(2).join("---").trim();

  // Extract metadata from frontmatter
  const metadata: any = {};
  frontmatter.split("\n").forEach((line: string) => {
    const trimmedLine = line.trim();
    if (trimmedLine && !trimmedLine.startsWith("input:") && !trimmedLine.startsWith("output:") && !trimmedLine.startsWith("schema:") && !trimmedLine.startsWith("format:")) {
      const colonIndex = trimmedLine.indexOf(":");
      if (colonIndex > 0) {
        const key = trimmedLine.substring(0, colonIndex).trim();
        const value = trimmedLine.substring(colonIndex + 1).trim();
        if (key === "temperature" || key === "max_tokens" || key === "top_p") {
          metadata[key] = parseFloat(value);
        } else if (key === "model") {
          metadata[key] = value;
        }
      }
    }
  });

  // Parse content to extract developer message and user messages
  let developerMessage = "";
  const messages: Message[] = [];
  const lines = content.split("\n");
  let currentRole: "user" | "assistant" | null = null;
  let currentContent = "";

  for (const line of lines) {
    if (line.startsWith("Developer:")) {
      developerMessage = line.substring("Developer:".length).trim();
    } else if (line.startsWith("User:")) {
      if (currentRole && currentContent) {
        messages.push({ role: currentRole, content: currentContent.trim() });
      }
      currentRole = "user";
      currentContent = line.substring("User:".length).trim();
    } else if (line.startsWith("Assistant:")) {
      if (currentRole && currentContent) {
        messages.push({ role: currentRole, content: currentContent.trim() });
      }
      currentRole = "assistant";
      currentContent = line.substring("Assistant:".length).trim();
    } else if (line.trim() && currentRole) {
      currentContent += "\n" + line.trim();
    }
  }

  // Add the last message
  if (currentRole && currentContent) {
    messages.push({ role: currentRole, content: currentContent.trim() });
  }

  // Parse tools from frontmatter if present
  const tools: Tool[] = [];
  // TODO: Add tool parsing if needed

  // Strip version suffix from prompt name for display
  const promptId = apiResponse?.prompt_spec?.prompt_id || "Unnamed Prompt";
  const baseName = stripVersionFromPromptId(promptId) || promptId;

  return {
    name: baseName,
    model: metadata.model || "gpt-4o",
    config: {
      temperature: metadata.temperature,
      max_tokens: metadata.max_tokens,
      top_p: metadata.top_p,
    },
    tools: tools,
    developerMessage: developerMessage,
    messages: messages.length > 0 ? messages : [{ role: "user", content: "Enter task specifics. Use {{template_variables}} for dynamic inputs" }],
  };
};

export const getVersionNumber = (promptId?: string): string => {
  if (!promptId) return "1";
  // Match version with dot (.v), underscore (_v), or hyphen (-v) separator
  const match = promptId.match(/[._-]v(\d+)$/);
  return match ? match[1] : "1";
};

export const stripVersionFromPromptId = (promptId?: string): string => {
  if (!promptId) return "";
  // Remove version suffix with dot (.v), underscore (_v), or hyphen (-v) separator
  return promptId.replace(/[._-]v\d+$/, "");
};
