import { EndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import { AUDIO_ACCEPT, CHAT_ATTACHMENT_ACCEPT, IMAGE_EDIT_ACCEPT } from "../chat_ui/uploadValidation";

export type AttachKind = "chat" | "images" | "audio";

export type Capabilities = Readonly<{
  target: "model" | "agent" | "mcpTool";
  mcp: boolean;
  vectorStores: boolean;
  guardrails: boolean;
  policies: boolean;
  attach: AttachKind | null;
  codeInterpreter: boolean;
  stream: boolean;
  simulateFallback: boolean;
  sampling: boolean;
  sessions: boolean;
  voice: boolean;
  customHeaders: boolean;
  textRequired: boolean;
  placeholder: string;
}>;

const BASE: Capabilities = {
  target: "model",
  mcp: false,
  vectorStores: false,
  guardrails: false,
  policies: false,
  attach: null,
  codeInterpreter: false,
  stream: false,
  simulateFallback: false,
  sampling: false,
  sessions: false,
  voice: false,
  customHeaders: true,
  textRequired: true,
  placeholder: "Send a test request",
};

const TEXT_CHAT: Partial<Capabilities> = {
  mcp: true,
  vectorStores: true,
  guardrails: true,
  policies: true,
  stream: true,
};

const CAPABILITIES: Readonly<Record<EndpointType, Capabilities>> = {
  [EndpointType.CHAT]: { ...BASE, ...TEXT_CHAT, attach: "chat", simulateFallback: true, sampling: true },
  [EndpointType.RESPONSES]: { ...BASE, ...TEXT_CHAT, attach: "chat", codeInterpreter: true, sessions: true },
  [EndpointType.ANTHROPIC_MESSAGES]: { ...BASE, ...TEXT_CHAT },
  [EndpointType.IMAGE]: { ...BASE, placeholder: "Describe the image you want to generate" },
  [EndpointType.VIDEO]: { ...BASE, placeholder: "Describe the video you want to generate" },
  [EndpointType.IMAGE_EDITS]: { ...BASE, attach: "images", placeholder: "Describe how to edit the images" },
  [EndpointType.EMBEDDINGS]: { ...BASE, placeholder: "Text to embed" },
  [EndpointType.SPEECH]: { ...BASE, voice: true, placeholder: "Text to turn into speech" },
  [EndpointType.TRANSCRIPTION]: {
    ...BASE,
    attach: "audio",
    textRequired: false,
    placeholder: "Optional context for the transcription",
  },
  [EndpointType.A2A_AGENTS]: { ...BASE, target: "agent", guardrails: true, placeholder: "Send a message to the agent" },
  [EndpointType.MCP]: { ...BASE, target: "mcpTool", guardrails: true, textRequired: false },
  [EndpointType.REALTIME]: { ...BASE, guardrails: true, customHeaders: false },
  [EndpointType.INTERACTIONS]: { ...BASE },
};

export const capabilitiesFor = (endpoint: EndpointType): Capabilities => CAPABILITIES[endpoint];

export const ACCEPT_FOR: Readonly<Record<AttachKind, string>> = {
  chat: CHAT_ATTACHMENT_ACCEPT,
  images: IMAGE_EDIT_ACCEPT,
  audio: AUDIO_ACCEPT,
};

export const ATTACH_LABEL: Readonly<Record<AttachKind, string>> = {
  chat: "Attach an image or PDF",
  images: "Add images to edit",
  audio: "Attach an audio file",
};

export const SUGGESTIONS: Readonly<Partial<Record<EndpointType, readonly string[]>>> = {
  [EndpointType.CHAT]: ["Write me a poem", "Explain quantum computing", "Draft a polite email requesting a meeting"],
  [EndpointType.RESPONSES]: [
    "Write me a poem",
    "Explain quantum computing",
    "Draft a polite email requesting a meeting",
  ],
  [EndpointType.ANTHROPIC_MESSAGES]: [
    "Write me a poem",
    "Explain quantum computing",
    "Draft a polite email requesting a meeting",
  ],
  [EndpointType.A2A_AGENTS]: ["What can you help me with?", "Tell me about yourself", "What tasks can you perform?"],
};

export const CODE_INTERPRETER_SUGGESTIONS: readonly string[] = [
  "Generate sample sales data CSV and create a chart",
  "Create a PNG bar chart comparing AI gateway providers including LiteLLM",
  "Generate a CSV of LLM pricing data and visualize it as a line chart",
];
