// 分类规则的固定说明保持独立，界面在使用处按语言翻译，配置值不变。
export type ClassificationRubric = "legacy" | "agentic" | "chat" | "business";
export const DEFAULT_CLASSIFICATION_RUBRIC: ClassificationRubric = "legacy";
export const NEW_CLASSIFIER_CLASSIFICATION_RUBRIC: ClassificationRubric = "agentic";

export const CLASSIFICATION_RUBRIC_DESCRIPTIONS: Record<ClassificationRubric, { label: string; description: string }> =
  {
    legacy: {
      label: "Legacy (uncalibrated)",
      description:
        "The rubric as it shipped before calibration examples, with no worked examples at all. Routers created " +
        "before this setting existed use it, so their tier decisions and spend are unchanged. It over-routes " +
        "ordinary engineering to the most expensive tier.",
    },
    agentic: {
      label: "Agentic",
      description:
        "Anchors routine installs, builds, multi-file edits, and standard debugging at " +
        "Medium, so ordinary engineering does not route to your most expensive tier. Suits agent, terminal, and " +
        "coding-assistant traffic, and mixed traffic.",
    },
    chat: {
      label: "Chat",
      description:
        "Drops the engineering examples, for a router serving only conversational traffic that never sees those " +
        "requests.",
    },
    business: {
      label: "Business",
      description:
        "Business and sales examples plus business-oriented tier definitions: routine drafting and summarizing " +
        "stay at Medium, data-determined analysis is Complex, and only decisions under conflicting tradeoffs " +
        "reach Reasoning. Suits sales, support, and go-to-market traffic.",
    },
  };

export const CLASSIFICATION_RUBRIC_KEYS = Object.keys(CLASSIFICATION_RUBRIC_DESCRIPTIONS) as ClassificationRubric[];
