import { EndpointType } from "./mode_endpoint_mapping";

export const OPEN_AI_VOICES = {
  ALLOY: "alloy",
  ASH: "ash",
  BALAD: "ballad",
  CORAL: "coral",
  ECHO: "echo",
  FABLE: "fable",
  NOVA: "nova",
  ONYX: "onyx",
  SAGE: "sage",
  SHIMMER: "shimmer",
} as const;

export type OpenAIVoice = (typeof OPEN_AI_VOICES)[keyof typeof OPEN_AI_VOICES];

export const OPEN_AI_VOICE_LABELS = {
  ALLOY: "Alloy - Professional and confident",
  ASH: "Ash - Casual and relaxed",
  BALAD: "Ballad - Smooth and melodic",
  CORAL: "Coral - Warm and engaging",
  ECHO: "Echo - Friendly and conversational",
  FABLE: "Fable - Wise and measured",
  NOVA: "Nova - Friendly and conversational",
  ONYX: "Onyx - Deep and authoritative",
  SAGE: "Sage - Wise and measured",
  SHIMMER: "Shimmer - Bright and cheerful",
};

export const OPEN_AI_VOICE_SELECT_OPTIONS = Object.entries(OPEN_AI_VOICES).map(([key, voice]) => ({
  value: voice,
  label: OPEN_AI_VOICE_LABELS[key as keyof typeof OPEN_AI_VOICE_LABELS],
}));

export const ENDPOINT_OPTIONS = [
  { value: EndpointType.CHAT, label: "/v1/chat/completions" },
  { value: EndpointType.RESPONSES, label: "/v1/responses" },
  { value: EndpointType.ANTHROPIC_MESSAGES, label: "/v1/messages" },
  { value: EndpointType.IMAGE, label: "/v1/images/generations" },
  { value: EndpointType.IMAGE_EDITS, label: "/v1/images/edits" },
  { value: EndpointType.EMBEDDINGS, label: "/v1/embeddings" },
  { value: EndpointType.SPEECH, label: "/v1/audio/speech" },
  { value: EndpointType.TRANSCRIPTION, label: "/v1/audio/transcriptions" },
  { value: EndpointType.A2A_AGENTS, label: "/v1/a2a/message/send" },
  { value: EndpointType.MCP, label: "/mcp-rest/tools/call" },
];
