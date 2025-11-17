import type {
  BedrockGuardrailResponse,
  BedrockAssessment,
  BedrockGuardrailCoverage,
  BedrockGuardrailUsage,
} from "@/components/view_logs/GuardrailViewer/BedrockGuardrailDetails";

export interface RecognitionMetadata {
  recognizer_name: string;
  recognizer_identifier: string;
}

export interface GuardrailEntity {
  end: number;
  score: number;
  start: number;
  entity_type: string;
  analysis_explanation: string | null;
  recognition_metadata: RecognitionMetadata;
}

export interface GuardrailInformation {
  duration: number;
  end_time: number;
  start_time: number;
  guardrail_mode: string;
  guardrail_name: string;
  guardrail_status: string;
  guardrail_response: GuardrailEntity[] | BedrockGuardrailResponse;
  masked_entity_count: Record<string, number>;
  guardrail_provider?: string;
}

// ===== Builders =====
export const makeEntity = (overrides: Partial<GuardrailEntity> = {}): GuardrailEntity => ({
  end: 18,
  start: 5,
  score: 0.92,
  entity_type: "EMAIL_ADDRESS",
  analysis_explanation: "Matched via pattern",
  recognition_metadata: {
    recognizer_name: "EmailRecognizer",
    recognizer_identifier: "email_v1",
  },
  ...overrides,
});

export const makeMaskedCounts = (overrides: Record<string, number> = {}) => ({
  EMAIL_ADDRESS: 2,
  PHONE_NUMBER: 1,
  ...overrides,
});

export const makeBedrockUsage = (overrides: Partial<BedrockGuardrailUsage> = {}): BedrockGuardrailUsage => ({
  contentPolicyUnits: 4,
  topicPolicyUnits: 2,
  ...overrides,
});

export const makeBedrockCoverage = (overrides: Partial<BedrockGuardrailCoverage> = {}): BedrockGuardrailCoverage => ({
  textCharacters: { guarded: 27, total: 100 },
  images: { guarded: 1, total: 3 },
  ...overrides,
});

export const makeAssessment = (overrides: Partial<BedrockAssessment> = {}): BedrockAssessment => ({
  wordPolicy: {
    customWords: [{ action: "BLOCKED", detected: true, match: "badword" }],
    managedWordLists: [{ action: "ALLOWED", detected: false, match: "ok", type: "PROFANITY" }],
  },
  contentPolicy: {
    filters: [
      { type: "HATE", action: "BLOCKED", detected: true, filterStrength: "HIGH", confidence: "MEDIUM" },
      { type: "VIOLENCE", action: "NONE", detected: false, filterStrength: "LOW", confidence: "LOW" },
    ],
  },
  topicPolicy: { topics: [{ name: "weapons", type: "DENY", detected: true, action: "BLOCKED" }] },
  sensitiveInformationPolicy: {
    piiEntities: [{ type: "EMAIL", match: "x@y.com", detected: true, action: "ANONYMIZED" }],
    regexes: [{ name: "ticket", regex: "#[0-9]+", match: "#123", detected: true, action: "BLOCKED" }],
  },
  contextualGroundingPolicy: {
    filters: [{ type: "GROUNDING", action: "BLOCKED", detected: true, score: 0.2, threshold: 0.5 }],
  },
  automatedReasoningPolicy: { findings: [{ foo: "bar" }] },
  invocationMetrics: {
    guardrailProcessingLatency: 42,
    usage: makeBedrockUsage(),
    guardrailCoverage: makeBedrockCoverage(),
  },
  ...overrides,
});

export const makeBedrockResponse = (overrides: Partial<BedrockGuardrailResponse> = {}): BedrockGuardrailResponse => ({
  action: "NONE",
  outputs: [{ text: "ok" }],
  usage: makeBedrockUsage(),
  guardrailCoverage: makeBedrockCoverage(),
  assessments: [makeAssessment()],
  ...overrides,
});

export const makeGuardrailInformation = (overrides: Partial<GuardrailInformation> = {}): GuardrailInformation => ({
  guardrail_name: "pii-rail",
  guardrail_mode: "post",
  guardrail_status: "success",
  start_time: 1_700_000_000,
  end_time: 1_700_000_123,
  duration: 0.123456,
  guardrail_response: [makeEntity()],
  masked_entity_count: makeMaskedCounts(),
  ...overrides,
});
