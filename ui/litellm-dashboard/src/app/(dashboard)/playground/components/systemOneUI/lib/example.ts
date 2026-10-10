import type { DecisionRequest, SystemOneRequest } from "./schemas";
import type { OpenAIDecisionsRequest } from "./openAIDecisions";

type PresetChoice = {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
};

type PresetNoul = {
  type: "noul";
  instructions: string;
  criteria?: { true: string; false: string };
};

type PresetScore = {
  type: "score";
  instructions: string;
  criteria: string[];
};

export type PresetQuestion = PresetChoice | PresetNoul | PresetScore;

export type PresetRequest = {
  state: string;
  questions: Record<string, PresetQuestion>;
};

type PresetEndpoint = "/v1/decisions" | "/v1/systemone";

export interface DecisionPreset {
  id: string;
  label: string;
  request: PresetRequest;
}

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

const { model: _exampleModel, ...bugTriage }: SystemOneRequest & PresetRequest = SYSTEM_ONE_EXAMPLE;

const supportRouting = {
  state:
    "Hi, I was charged twice for my team plan this month and the second charge pushed my card over its limit. I need the duplicate refunded today and want to know why it happened.",
  questions: {
    department: {
      type: "choice",
      instructions: "Which team should handle this ticket?",
      criteria: {
        billing: "Invoices, charges, refunds, and plan changes",
        technical: "Bugs, errors, outages, and integration help",
        account: "Login, permissions, and profile settings",
        sales: "Pricing questions and new purchases",
      },
    },
  },
} satisfies PresetRequest;

const moderation = {
  state:
    "This guide is garbage and whoever wrote it should be fired. The fix that actually works is setting stream=false, took me two days to figure that out.",
  questions: {
    breaks_rules: {
      type: "noul",
      instructions: "Does this comment break the community guidelines?",
      criteria: {
        true: "Insults, harassment, slurs, or threats aimed at a person",
        false: "Blunt or frustrated but still about the content",
      },
    },
  },
} satisfies PresetRequest;

const leadScoring = {
  state:
    "Inbound form: VP Engineering at a 400 person fintech. Running a self hosted gateway today, evaluating replacements this quarter, budget approved, needs SSO and audit logs.",
  questions: {
    fit: {
      type: "score",
      instructions: "How well does this lead fit our ideal customer?",
      criteria: [
        "No fit, wrong market or no budget",
        "Weak fit, early curiosity only",
        "Possible fit, needs discovery",
        "Strong fit, active evaluation",
        "Ideal fit, ready to buy this quarter",
      ],
    },
  },
} satisfies PresetRequest;

export const DECISION_PRESETS: readonly DecisionPreset[] = [
  { id: "bug-triage", label: "Triage a bug report", request: bugTriage },
  { id: "support-routing", label: "Route a support ticket", request: supportRouting },
  { id: "moderation", label: "Moderate a forum comment", request: moderation },
  { id: "lead-scoring", label: "Score a sales lead", request: leadScoring },
];

const toOpenAIQuestion = (name: string, question: PresetQuestion): OpenAIDecisionsRequest["questions"][number] => {
  switch (question.type) {
    case "choice":
      return {
        type: "choice",
        name,
        instructions: question.instructions,
        choices: Object.entries(question.criteria).map(([value, description]) => ({ value, description })),
      };
    case "noul":
      return { type: "predicate", name, instructions: question.instructions };
    case "score":
      return {
        type: "score",
        name,
        instructions: question.instructions,
        levels: question.criteria.map((description, index) => ({ label: String(index), description })),
      };
  }
};

export const toSystemOneRequest = (request: PresetRequest, model?: string): DecisionRequest => ({ ...request, model });

export const toOpenAIDecisionsRequest = (request: PresetRequest, model?: string): OpenAIDecisionsRequest => ({
  model,
  input: request.state,
  questions: Object.entries(request.questions).map(([name, question]) => toOpenAIQuestion(name, question)),
});

export const presetPayload = (preset: DecisionPreset, endpoint: PresetEndpoint, model?: string): string =>
  JSON.stringify(
    endpoint === "/v1/decisions"
      ? toOpenAIDecisionsRequest(preset.request, model)
      : toSystemOneRequest(preset.request, model),
    null,
    2,
  );

export const emptyPayload = (endpoint: PresetEndpoint, model?: string): string =>
  JSON.stringify(
    endpoint === "/v1/decisions" ? { model, input: "", questions: [] } : { model, state: "", questions: {} },
    null,
    2,
  );

export const decisionsExample = (model?: string): DecisionRequest => toSystemOneRequest(bugTriage, model);

export const openAIDecisionsExample = (model?: string): OpenAIDecisionsRequest =>
  toOpenAIDecisionsRequest(bugTriage, model);
