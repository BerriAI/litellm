import type { TFunction } from "i18next";

export const getTestModes = (t: TFunction) => [
  { value: "chat", label: t("addModel.addModelModes.chatLabel") },
  { value: "completion", label: t("addModel.addModelModes.completionLabel") },
  { value: "embedding", label: t("addModel.addModelModes.embeddingLabel") },
  { value: "audio_speech", label: t("addModel.addModelModes.audioSpeechLabel") },
  { value: "audio_transcription", label: t("addModel.addModelModes.audioTranscriptionLabel") },
  { value: "image_generation", label: t("addModel.addModelModes.imageGenerationLabel") },
  { value: "video_generation", label: t("addModel.addModelModes.videoGenerationLabel") },
  { value: "rerank", label: t("addModel.addModelModes.rerankLabel") },
  { value: "realtime", label: t("addModel.addModelModes.realtimeLabel") },
  { value: "batch", label: t("addModel.addModelModes.batchLabel") },
  { value: "ocr", label: t("addModel.addModelModes.ocrLabel") },
];

export const getAutoRouterModes = (t: TFunction) => [
  { value: "simple-shuffle", label: t("addModel.addModelModes.simpleShuffleLabel") },
  { value: "least-busy", label: t("addModel.addModelModes.leastBusyLabel") },
  { value: "latency-based", label: t("addModel.addModelModes.latencyBasedLabel") },
  { value: "cost-based", label: t("addModel.addModelModes.costBasedLabel") },
  { value: "usage-based", label: t("addModel.addModelModes.usageBasedLabel") },
  { value: "custom", label: t("addModel.addModelModes.customLabel") },
];
