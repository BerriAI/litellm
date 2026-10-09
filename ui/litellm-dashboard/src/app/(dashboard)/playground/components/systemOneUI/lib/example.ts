import type { DecisionRequest, SystemOneRequest } from "./schemas";

export const SYSTEM_ONE_EXAMPLE = {
  model: "jev-latest",
  state:
    "Since upgrading to the latest release, streaming responses stop halfway through whenever a fallback model takes over. Non-streaming requests still work. I haven't narrowed down which change caused it, but it happens on most long prompts.",
  questions: {
    area: {
      type: "choice",
      instructions: "Which part of the project does this issue belong to?",
      criteria: {
        backend: "Server, API routing, streaming, and fallbacks",
        sdk: "The client library and its helpers",
        ui: "The web dashboard",
        docs: "Documentation and examples",
      },
    },
    has_repro_steps: {
      type: "noul",
      instructions: "Does the issue include steps someone could follow to reproduce it?",
      criteria: {
        true: "It gives concrete steps, a config, or a request that triggers the bug",
        false: "It only describes the symptom without a way to trigger it",
      },
    },
    severity: {
      type: "score",
      instructions: "How severe is this issue?",
      criteria: [
        "Cosmetic or a typo",
        "Minor bug with an easy workaround",
        "Broken feature with a workaround",
        "Broken feature with no workaround",
        "Outage or data loss",
      ],
    },
  },
} satisfies SystemOneRequest;

export const DECISIONS_EXAMPLE: DecisionRequest = {
  ...SYSTEM_ONE_EXAMPLE,
  model: "your-decision-model",
};
