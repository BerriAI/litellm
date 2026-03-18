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

type ParsedFrontmatter = {
  model?: string;
  config: {
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  };
  tools: Tool[];
};

const parseNumber = (raw: string): number | undefined => {
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
};

const parseToolsFromFrontmatter = (lines: string[]): Tool[] => {
  const tools: Tool[] = [];
  let inToolsBlock = false;

  for (const line of lines) {
    const trimmed = line.trim();

    if (!inToolsBlock) {
      if (trimmed === "tools:" || trimmed.startsWith("tools:")) {
        inToolsBlock = true;
      }
      continue;
    }

    // New top-level key ends the tools block
    if (line.length > 0 && !/^\s/.test(line) && trimmed !== "-" && !trimmed.startsWith("-")) {
      break;
    }

    const match = trimmed.match(/^-+\s*(.+)$/);
    if (!match) continue;

    const rawJson = match[1].trim();
    if (!rawJson) continue;

    try {
      const toolObj = JSON.parse(rawJson);
      tools.push({
        name: toolObj?.function?.name || "Unnamed Tool",
        description: toolObj?.function?.description || "",
        json: JSON.stringify(toolObj, null, 2),
      });
    } catch {
    }
  }

  return tools;
};

const parseDotpromptFrontmatter = (frontmatter: string): ParsedFrontmatter => {
  const result: ParsedFrontmatter = { config: {}, tools: [] };
  const lines = frontmatter.split("\n");

  result.tools = parseToolsFromFrontmatter(lines);

  for (const line of lines) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;

    // Skip known nested yaml sections and list items.
    if (
      trimmedLine.startsWith("input:") ||
      trimmedLine.startsWith("output:") ||
      trimmedLine.startsWith("schema:") ||
      trimmedLine.startsWith("format:") ||
      trimmedLine.startsWith("tools:") ||
      trimmedLine.startsWith("-")
    ) {
      continue;
    }

    const colonIndex = trimmedLine.indexOf(":");
    if (colonIndex <= 0) continue;

    const key = trimmedLine.substring(0, colonIndex).trim();
    const value = trimmedLine.substring(colonIndex + 1).trim();

    if (key === "model") {
      result.model = value;
      continue;
    }

    if (key === "temperature") result.config.temperature = parseNumber(value);
    if (key === "max_tokens") result.config.max_tokens = parseNumber(value);
    if (key === "top_p") result.config.top_p = parseNumber(value);
  }

  return result;
};

type ParsedBody = { developerMessage: string; messages: Message[] };

const parseDotpromptBody = (body: string): ParsedBody => {
  const roleHeader = /^(System|Developer|User|Assistant):(?:\s(.*)|\s*)$/;
  const messages: Message[] = [];
  let developerMessage = "";

  let currentRole: string | null = null;
  let buffer: string[] = [];

  const commit = () => {
    if (!currentRole) return;

    const content = buffer.join("\n").trim();
    if (currentRole === "developer") {
      if (content) {
        developerMessage = developerMessage ? `${developerMessage}\n\n${content}` : content;
      }
    } else if (content) {
      messages.push({ role: currentRole, content });
    } else {
      messages.push({ role: currentRole, content: "" });
    }
  };

  for (const line of body.split("\n")) {
    const match = line.match(roleHeader);
    if (match) {
      commit();
      currentRole = match[1].toLowerCase();
      buffer = [match[2] ?? ""];
      continue;
    }

    if (!currentRole) continue;
    buffer.push(line);
  }

  commit();

  return { developerMessage, messages };
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

  const frontmatter = parts[1];
  const content = parts.slice(2).join("---").trim();

  const parsedFrontmatter = parseDotpromptFrontmatter(frontmatter);
  const parsedBody = parseDotpromptBody(content);

  // Strip version suffix from prompt name for display
  const promptId = apiResponse?.prompt_spec?.prompt_id || "Unnamed Prompt";
  const baseName = stripVersionFromPromptId(promptId) || promptId;

  return {
    name: baseName,
    model: parsedFrontmatter.model || "gpt-4o",
    config: parsedFrontmatter.config,
    tools: parsedFrontmatter.tools,
    developerMessage: parsedBody.developerMessage,
    messages:
      parsedBody.messages.length > 0
        ? parsedBody.messages
        : [{ role: "user", content: "Enter task specifics. Use {{template_variables}} for dynamic inputs" }],
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
