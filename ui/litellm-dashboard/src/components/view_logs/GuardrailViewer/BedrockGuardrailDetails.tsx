import React, { useState } from "react";

export type BedrockGuardrailAction = "NONE" | "GUARDRAIL_INTERVENED";

export interface BedrockGuardrailUsage {
  wordPolicyUnits?: number;
  topicPolicyUnits?: number;
  contentPolicyUnits?: number;
  contentPolicyImageUnits?: number;
  automatedReasoningPolicies?: number;
  automatedReasoningPolicyUnits?: number;
  contextualGroundingPolicyUnits?: number;
  sensitiveInformationPolicyUnits?: number;
  sensitiveInformationPolicyFreeUnits?: number;
}

export interface BedrockGuardrailCoverageDim {
  total?: number;
  guarded?: number;
}

export interface BedrockGuardrailCoverage {
  images?: BedrockGuardrailCoverageDim;
  textCharacters?: BedrockGuardrailCoverageDim;
}

export interface BedrockWordItem {
  action?: string; // e.g., "BLOCKED" or "ALLOWED"
  detected?: boolean;
  match?: string;
  type?: string; // present on managedWordLists entries
}

export interface BedrockContentFilter {
  type?: string; // e.g. "HATE", etc.
  action?: string; // "BLOCKED" | "NONE" (varies by config)
  detected?: boolean;
  filterStrength?: string; // e.g., "HIGH"/"MEDIUM"/"LOW"
  confidence?: string; // e.g., "HIGH"/"MEDIUM"/"LOW"
}

export interface BedrockContextualGroundingFilter {
  type?: string; // "GROUNDING" | "RELEVANCE"
  action?: string; // "BLOCKED" | ...
  detected?: boolean;
  score?: number; // model score
  threshold?: number; // configured threshold
}

export interface BedrockPiiEntity {
  type?: string; // e.g., "ADDRESS", "EMAIL", etc.
  match?: string; // the matched text (may be masked)
  detected?: boolean;
  action?: string; // "BLOCKED" | "ANONYMIZED" | ...
}

export interface BedrockRegexFinding {
  name?: string;
  regex?: string;
  match?: string;
  detected?: boolean;
  action?: string;
}

export interface BedrockTopic {
  name?: string; // topic list entry name
  type?: string; // "DENY" | "ALLOW" depending on list
  detected?: boolean;
  action?: string; // "BLOCKED" | "NONE"
}

export interface BedrockInvocationMetrics {
  usage?: BedrockGuardrailUsage;
  guardrailCoverage?: BedrockGuardrailCoverage;
  guardrailProcessingLatency?: number; // in ms
}

export interface BedrockAssessment {
  wordPolicy?: { customWords?: BedrockWordItem[]; managedWordLists?: BedrockWordItem[] };
  contentPolicy?: { filters?: BedrockContentFilter[] };
  topicPolicy?: { topics?: BedrockTopic[] };
  sensitiveInformationPolicy?: { piiEntities?: BedrockPiiEntity[]; regexes?: BedrockRegexFinding[] };
  contextualGroundingPolicy?: { filters?: BedrockContextualGroundingFilter[] };
  automatedReasoningPolicy?: { findings?: any[] };
  invocationMetrics?: BedrockInvocationMetrics;
}

export interface BedrockOutputContent {
  text?: string;
}

export interface BedrockGuardrailResponse {
  action?: BedrockGuardrailAction;
  actionReason?: string | null;
  outputs?: BedrockOutputContent[];
  output?: BedrockOutputContent[];
  usage?: BedrockGuardrailUsage;
  guardrailCoverage?: BedrockGuardrailCoverage;
  assessments?: BedrockAssessment[];
  blockedResponse?: string;
}

/** ====== UI helpers ====== */
type ChipTone = "green" | "red" | "blue" | "slate" | "amber";

const chip = (text: React.ReactNode, tone: ChipTone = "slate") => {
  const map: Record<ChipTone, string> = {
    green: "bg-green-100 text-green-800",
    red: "bg-red-100 text-red-800",
    blue: "bg-blue-50 text-blue-700",
    slate: "bg-slate-100 text-slate-800",
    amber: "bg-amber-100 text-amber-800",
  };
  return <span className={`px-2 py-1 rounded-md text-xs font-medium inline-block ${map[tone]}`}>{text}</span>;
};

const boolPill = (b?: boolean) => (b ? chip("detected", "red") : chip("not detected", "slate"));

interface SectionProps {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  right?: React.ReactNode;
  children?: React.ReactNode;
}

const Section: React.FC<SectionProps> = ({ title, count, defaultOpen = true, right, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border rounded-lg overflow-hidden">
      <div
        className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center">
          <svg
            className={`w-5 h-5 mr-2 transition-transform ${open ? "transform rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h5 className="font-medium">
            {title} {typeof count === "number" && <span className="text-gray-500 font-normal">({count})</span>}
          </h5>
        </div>
        <div>{right}</div>
      </div>
      {open && <div className="p-3 border-t bg-white">{children}</div>}
    </div>
  );
};

interface KVProps {
  label: string;
  children?: React.ReactNode;
  mono?: boolean;
}

const KV: React.FC<KVProps> = ({ label, children, mono }) => (
  <div className="flex">
    <span className="font-medium w-1/3">{label}</span>
    <span className={mono ? "font-mono text-sm break-all" : ""}>{children}</span>
  </div>
);

const Divider: React.FC = () => <div className="my-3 border-t" />;

/** ====== Main component ====== */
export const BedrockGuardrailDetails: React.FC<{ response: BedrockGuardrailResponse }> = ({ response }) => {
  if (!response) return null;

  const outputs: BedrockOutputContent[] = (response.outputs ?? response.output ?? []) as BedrockOutputContent[];

  const actionTone: "green" | "red" = response.action === "GUARDRAIL_INTERVENED" ? "red" : "green";

  const coverageChips = (
    <div className="flex flex-wrap gap-2">
      {response.guardrailCoverage?.textCharacters &&
        chip(
          `text guarded ${response.guardrailCoverage.textCharacters.guarded ?? 0}/${response.guardrailCoverage.textCharacters.total ?? 0}`,
          "blue",
        )}
      {response.guardrailCoverage?.images &&
        chip(
          `images guarded ${response.guardrailCoverage.images.guarded ?? 0}/${response.guardrailCoverage.images.total ?? 0}`,
          "blue",
        )}
    </div>
  );

  const usagePills = response.usage && (
    <div className="flex flex-wrap gap-2">
      {Object.entries(response.usage).map(([k, v]) =>
        typeof v === "number" ? (
          <span key={k} className="px-2 py-1 bg-slate-100 text-slate-800 rounded-md text-xs font-medium">
            {k}: {v}
          </span>
        ) : null,
      )}
    </div>
  );

  return (
    <div className="space-y-3">
      {/* Top summary card */}
      <div className="border rounded-lg p-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <KV label="Action:">{chip(response.action ?? "N/A", actionTone)}</KV>
            {response.actionReason && <KV label="Action Reason:">{response.actionReason}</KV>}
            {response.blockedResponse && (
              <KV label="Blocked Response:">
                <span className="italic">{response.blockedResponse}</span>
              </KV>
            )}
          </div>
          <div className="space-y-2">
            <KV label="Coverage:">{coverageChips}</KV>
            <KV label="Usage:">{usagePills}</KV>
          </div>
        </div>

        {/* Outputs */}
        {outputs.length > 0 && (
          <>
            <Divider />
            <h4 className="font-medium mb-2">Outputs</h4>
            <div className="space-y-2">
              {outputs.map((o, i) => (
                <div key={i} className="p-3 bg-gray-50 rounded-md">
                  <div className="text-sm whitespace-pre-wrap">{o.text ?? <em>(non-text output)</em>}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      {/* Assessments */}
      {response.assessments?.length ? (
        <div className="space-y-3">
          {response.assessments.map((assess, idx) => {
            const policyBadges = (
              <div className="flex flex-wrap gap-1">
                {assess.wordPolicy && chip("word", "slate")}
                {assess.contentPolicy && chip("content", "slate")}
                {assess.topicPolicy && chip("topic", "slate")}
                {assess.sensitiveInformationPolicy && chip("sensitive-info", "slate")}
                {assess.contextualGroundingPolicy && chip("contextual-grounding", "slate")}
                {assess.automatedReasoningPolicy && chip("automated-reasoning", "slate")}
              </div>
            );

            return (
              <Section
                key={idx}
                title={`Assessment #${idx + 1}`}
                defaultOpen
                right={
                  <div className="flex items-center gap-3">
                    {assess.invocationMetrics?.guardrailProcessingLatency != null &&
                      chip(`${assess.invocationMetrics.guardrailProcessingLatency} ms`, "amber")}
                    {policyBadges}
                  </div>
                }
              >
                {/* Word policy */}
                {assess.wordPolicy && (
                  <div className="mb-3">
                    <h6 className="font-medium mb-2">Word Policy</h6>
                    {(assess.wordPolicy.customWords?.length ?? 0) > 0 && (
                      <Section title="Custom Words" defaultOpen>
                        <div className="space-y-2">
                          {assess.wordPolicy.customWords!.map((w, i) => (
                            <div key={i} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <div className="flex items-center gap-2">
                                {chip(w.action ?? "N/A", w.detected ? "red" : "slate")}
                                <span className="font-mono text-sm break-all">{w.match}</span>
                              </div>
                              {boolPill(w.detected)}
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}
                    {(assess.wordPolicy.managedWordLists?.length ?? 0) > 0 && (
                      <Section title="Managed Word Lists" defaultOpen={false}>
                        <div className="space-y-2">
                          {assess.wordPolicy.managedWordLists!.map((w, i) => (
                            <div key={i} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <div className="flex items-center gap-2">
                                {chip(w.action ?? "N/A", w.detected ? "red" : "slate")}
                                <span className="font-mono text-sm break-all">{w.match}</span>
                                {w.type && chip(w.type, "slate")}
                              </div>
                              {boolPill(w.detected)}
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}
                  </div>
                )}

                {/* Content policy */}
                {assess.contentPolicy?.filters?.length ? (
                  <div className="mb-3">
                    <h6 className="font-medium mb-2">Content Policy</h6>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="text-left text-gray-600">
                            <th className="py-1 pr-4">Type</th>
                            <th className="py-1 pr-4">Action</th>
                            <th className="py-1 pr-4">Detected</th>
                            <th className="py-1 pr-4">Strength</th>
                            <th className="py-1 pr-4">Confidence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {assess.contentPolicy.filters!.map((f, i) => (
                            <tr key={i} className="border-t">
                              <td className="py-1 pr-4">{f.type ?? "—"}</td>
                              <td className="py-1 pr-4">{chip(f.action ?? "—", f.detected ? "red" : "slate")}</td>
                              <td className="py-1 pr-4">{boolPill(f.detected)}</td>
                              <td className="py-1 pr-4">{f.filterStrength ?? "—"}</td>
                              <td className="py-1 pr-4">{f.confidence ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                {/* Contextual grounding */}
                {assess.contextualGroundingPolicy?.filters?.length ? (
                  <div className="mb-3">
                    <h6 className="font-medium mb-2">Contextual Grounding</h6>
                    <div className="overflow-x-auto">
                      <table className="min-w-full text-sm">
                        <thead>
                          <tr className="text-left text-gray-600">
                            <th className="py-1 pr-4">Type</th>
                            <th className="py-1 pr-4">Action</th>
                            <th className="py-1 pr-4">Detected</th>
                            <th className="py-1 pr-4">Score</th>
                            <th className="py-1 pr-4">Threshold</th>
                          </tr>
                        </thead>
                        <tbody>
                          {assess.contextualGroundingPolicy.filters!.map((f, i) => (
                            <tr key={i} className="border-t">
                              <td className="py-1 pr-4">{f.type ?? "—"}</td>
                              <td className="py-1 pr-4">{chip(f.action ?? "—", f.detected ? "red" : "slate")}</td>
                              <td className="py-1 pr-4">{boolPill(f.detected)}</td>
                              <td className="py-1 pr-4">{f.score ?? "—"}</td>
                              <td className="py-1 pr-4">{f.threshold ?? "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                {/* Sensitive Information */}
                {assess.sensitiveInformationPolicy && (
                  <div className="mb-3">
                    <h6 className="font-medium mb-2">Sensitive Information</h6>
                    {(assess.sensitiveInformationPolicy.piiEntities?.length ?? 0) > 0 && (
                      <Section title="PII Entities" defaultOpen>
                        <div className="space-y-2">
                          {assess.sensitiveInformationPolicy.piiEntities!.map((p, i) => (
                            <div key={i} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                              <div className="flex items-center gap-2">
                                {chip(p.action ?? "N/A", p.detected ? "red" : "slate")}
                                {p.type && chip(p.type, "slate")}
                                <span className="font-mono text-xs break-all">{p.match}</span>
                              </div>
                              {boolPill(p.detected)}
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}
                    {(assess.sensitiveInformationPolicy.regexes?.length ?? 0) > 0 && (
                      <Section title="Custom Regexes" defaultOpen={false}>
                        <div className="space-y-2">
                          {assess.sensitiveInformationPolicy.regexes!.map((r, i) => (
                            <div
                              key={i}
                              className="flex flex-col sm:flex-row sm:items-center sm:justify-between p-2 bg-gray-50 rounded gap-1"
                            >
                              <div className="flex items-center gap-2">
                                {chip(r.action ?? "N/A", r.detected ? "red" : "slate")}
                                <span className="font-medium">{r.name ?? "regex"}</span>
                                <span className="font-mono text-xs break-all">{r.regex}</span>
                              </div>
                              <div className="flex items-center gap-2">
                                {boolPill(r.detected)}
                                {r.match && <span className="font-mono text-xs break-all">{r.match}</span>}
                              </div>
                            </div>
                          ))}
                        </div>
                      </Section>
                    )}
                  </div>
                )}

                {/* Topic policy */}
                {assess.topicPolicy?.topics?.length ? (
                  <div className="mb-3">
                    <h6 className="font-medium mb-2">Topic Policy</h6>
                    <div className="flex flex-wrap gap-2">
                      {assess.topicPolicy.topics!.map((t, i) => (
                        <div key={i} className="px-3 py-1.5 bg-gray-50 rounded-md text-xs">
                          <div className="flex items-center gap-2">
                            {chip(t.action ?? "N/A", t.detected ? "red" : "slate")}
                            <span className="font-medium">{t.name ?? "topic"}</span>
                            {t.type && chip(t.type, "slate")}
                            {boolPill(t.detected)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : null}

                {/* Invocation metrics */}
                {assess.invocationMetrics && (
                  <Section title="Invocation Metrics" defaultOpen={false}>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <KV label="Latency (ms)">{assess.invocationMetrics.guardrailProcessingLatency ?? "—"}</KV>
                        <KV label="Coverage:">
                          <div className="flex flex-wrap gap-2">
                            {assess.invocationMetrics.guardrailCoverage?.textCharacters &&
                              chip(
                                `text ${assess.invocationMetrics.guardrailCoverage.textCharacters.guarded ?? 0}/${
                                  assess.invocationMetrics.guardrailCoverage.textCharacters.total ?? 0
                                }`,
                                "blue",
                              )}
                            {assess.invocationMetrics.guardrailCoverage?.images &&
                              chip(
                                `images ${assess.invocationMetrics.guardrailCoverage.images.guarded ?? 0}/${
                                  assess.invocationMetrics.guardrailCoverage.images.total ?? 0
                                }`,
                                "blue",
                              )}
                          </div>
                        </KV>
                      </div>
                      <div className="space-y-2">
                        <KV label="Usage:">
                          <div className="flex flex-wrap gap-2">
                            {assess.invocationMetrics.usage &&
                              Object.entries(assess.invocationMetrics.usage).map(([k, v]) =>
                                typeof v === "number" ? (
                                  <span
                                    key={k}
                                    className="px-2 py-1 bg-slate-100 text-slate-800 rounded-md text-xs font-medium"
                                  >
                                    {k}: {v}
                                  </span>
                                ) : null,
                              )}
                          </div>
                        </KV>
                      </div>
                    </div>
                  </Section>
                )}

                {/* Automated reasoning (fallback render) */}
                {assess.automatedReasoningPolicy?.findings?.length ? (
                  <Section title="Automated Reasoning Findings" defaultOpen={false}>
                    <div className="space-y-2">
                      {assess.automatedReasoningPolicy.findings!.map((f, i) => (
                        <pre key={i} className="bg-gray-50 rounded p-2 text-xs overflow-x-auto">
                          {JSON.stringify(f, null, 2)}
                        </pre>
                      ))}
                    </div>
                  </Section>
                ) : null}
              </Section>
            );
          })}
        </div>
      ) : null}

      {/* Raw JSON (for debugging / completeness) */}
      <Section title="Raw Bedrock Guardrail Response" defaultOpen={false}>
        <pre className="bg-gray-50 rounded p-3 text-xs overflow-x-auto">{JSON.stringify(response, null, 2)}</pre>
      </Section>
    </div>
  );
};

export default BedrockGuardrailDetails;
