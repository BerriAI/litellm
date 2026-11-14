import type { ParseKeys } from "i18next";
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
  placeholder: ParseKeys;
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
  placeholder: "playground.chat.endpointCapabilities.placeholders.default",
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
  [EndpointType.IMAGE]: { ...BASE, placeholder: "playground.chat.endpointCapabilities.placeholders.image" },
  [EndpointType.VIDEO]: { ...BASE, placeholder: "playground.chat.endpointCapabilities.placeholders.video" },
  [EndpointType.IMAGE_EDITS]: {
    ...BASE,
    attach: "images",
    placeholder: "playground.chat.endpointCapabilities.placeholders.imageEdits",
  },
  [EndpointType.EMBEDDINGS]: { ...BASE, placeholder: "playground.chat.endpointCapabilities.placeholders.embeddings" },
  [EndpointType.SPEECH]: {
    ...BASE,
    voice: true,
    placeholder: "playground.chat.endpointCapabilities.placeholders.speech",
  },
  [EndpointType.TRANSCRIPTION]: {
    ...BASE,
    attach: "audio",
    textRequired: false,
    placeholder: "playground.chat.endpointCapabilities.placeholders.transcription",
  },
  [EndpointType.A2A_AGENTS]: {
    ...BASE,
    target: "agent",
    guardrails: true,
    placeholder: "playground.chat.endpointCapabilities.placeholders.agent",
  },
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

export const ATTACH_LABEL: Readonly<Record<AttachKind, ParseKeys>> = {
  chat: "playground.chat.endpointCapabilities.attach.chat",
  images: "playground.chat.endpointCapabilities.attach.images",
  audio: "playground.chat.endpointCapabilities.attach.audio",
};

export const SUGGESTIONS: Readonly<Partial<Record<EndpointType, readonly ParseKeys[]>>> = {
  [EndpointType.CHAT]: [
    "playground.chat.endpointCapabilities.suggestions.poem",
    "playground.chat.endpointCapabilities.suggestions.quantum",
    "playground.chat.endpointCapabilities.suggestions.email",
  ],
  [EndpointType.RESPONSES]: [
    "playground.chat.endpointCapabilities.suggestions.poem",
    "playground.chat.endpointCapabilities.suggestions.quantum",
    "playground.chat.endpointCapabilities.suggestions.email",
  ],
  [EndpointType.ANTHROPIC_MESSAGES]: [
    "playground.chat.endpointCapabilities.suggestions.poem",
    "playground.chat.endpointCapabilities.suggestions.quantum",
    "playground.chat.endpointCapabilities.suggestions.email",
  ],
  [EndpointType.A2A_AGENTS]: [
    "playground.chat.endpointCapabilities.suggestions.agentHelp",
    "playground.chat.endpointCapabilities.suggestions.agentAbout",
    "playground.chat.endpointCapabilities.suggestions.agentTasks",
  ],
};

export const CODE_INTERPRETER_SUGGESTIONS: readonly ParseKeys[] = [
  "playground.chat.endpointCapabilities.codeInterpreterSuggestions.salesChart",
  "playground.chat.endpointCapabilities.codeInterpreterSuggestions.providersChart",
  "playground.chat.endpointCapabilities.codeInterpreterSuggestions.pricingChart",
];
