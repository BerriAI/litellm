import type { ParseKeys } from "i18next";

// Define the available test modes
export const TEST_MODES: readonly { value: string; labelKey: ParseKeys }[] = [
  { value: "chat", labelKey: "addModel.addModelModes.chat" },
  { value: "completion", labelKey: "addModel.addModelModes.completion" },
  { value: "embedding", labelKey: "addModel.addModelModes.embedding" },
  { value: "audio_speech", labelKey: "addModel.addModelModes.audioSpeech" },
  { value: "audio_transcription", labelKey: "addModel.addModelModes.audioTranscription" },
  { value: "image_generation", labelKey: "addModel.addModelModes.imageGeneration" },
  { value: "image_edit", labelKey: "addModel.addModelModes.imageEdit" },
  { value: "video_generation", labelKey: "addModel.addModelModes.videoGeneration" },
  { value: "rerank", labelKey: "addModel.addModelModes.rerank" },
  { value: "realtime", labelKey: "addModel.addModelModes.realtime" },
  { value: "batch", labelKey: "addModel.addModelModes.batch" },
  { value: "anthropic_messages", labelKey: "addModel.addModelModes.anthropicMessages" },
  { value: "ocr", labelKey: "addModel.addModelModes.ocr" },
  { value: "decisions", labelKey: "addModel.addModelModes.decisions" },
  { value: "extraction", labelKey: "addModel.addModelModes.extraction" },
];

// Define the available auto router routing strategies
export const AUTO_ROUTER_MODES = [
  { value: "simple-shuffle", label: "Simple Shuffle - Random selection from available models" },
  { value: "least-busy", label: "Least Busy - Route to model with lowest current load" },
  { value: "latency-based", label: "Latency Based - Route to model with best response time" },
  { value: "cost-based", label: "Cost Based - Route to most cost-effective model" },
  { value: "usage-based", label: "Usage Based - Route based on historical usage patterns" },
  { value: "custom", label: "Custom - Use custom routing logic defined in config" },
];
