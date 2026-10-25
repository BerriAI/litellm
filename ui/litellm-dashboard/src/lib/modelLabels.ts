import type { Translate } from "@/i18n/language";

export type ModelLabelKind = "mode" | "feature";

// 以所属字段的固定枚举决定是否翻译，扩展值即使与其他界面文案同名也保留原展示。
const MODEL_LABELS: Readonly<Record<ModelLabelKind, ReadonlySet<string>>> = {
  mode: new Set([
    "chat",
    "embedding",
    "completion",
    "image_generation",
    "image_edit",
    "audio_transcription",
    "audio_speech",
    "rerank",
    "realtime",
    "decisions",
  ]),
  feature: new Set([
    "Parallel Function Calling",
    "Function Calling",
    "Vision",
    "System Messages",
    "Tool Choice",
    "Reasoning",
    "Response Schema",
    "Prompt Caching",
  ]),
};

export const translateModelLabel = (value: string, kind: ModelLabelKind, t: Translate, displayLabel = value): string =>
  MODEL_LABELS[kind].has(value) ? t(displayLabel) : displayLabel;
