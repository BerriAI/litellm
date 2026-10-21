import { ApiError } from "@/lib/http/client";
import type { SpanDetail } from "@/components/lens/traces/types";

import { evidenceTarget } from "../../model/findings";
import type { DatasetsApi } from "../../datasets/client";
import type { BuildSource, CaseSource, Dataset, DatasetCase, DatasetMessage, SkippedCase } from "../../datasets/types";
import type { LensDemoData } from "./fixtures";

const MAX_CASES = 200;
const NO_SOURCE: CaseSource = { trace_id: "", trace_ref: "", span_id: "", finding_id: "", lens_id: "" };
const ROLES: ReadonlySet<string> = new Set(["system", "user", "assistant", "tool"]);

type Candidate = DatasetCase | SkippedCase;

const isRole = (role: unknown): role is DatasetMessage["role"] => typeof role === "string" && ROLES.has(role);

function contentHash(text: string): string {
  const hash = [...text].reduce((acc, char) => (Math.imul(acc, 31) + char.charCodeAt(0)) | 0, 7);
  return (hash >>> 0).toString(16).padStart(8, "0");
}

function parsedMessages(raw: string): DatasetMessage[] {
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.flatMap((item: { role?: unknown; content?: unknown }) =>
      isRole(item?.role) && typeof item.content === "string"
        ? [{ role: item.role, content: item.content, name: "", tool_calls: [] }]
        : [],
    );
  } catch {
    return [];
  }
}

function makeCase(messages: DatasetMessage[], reply: string, source: CaseSource): Candidate {
  if (messages.length === 0 && !reply) return { source, reason: "no_content" };
  const id = contentHash(JSON.stringify({ messages, reply }));
  return { id, messages, reply, tool_calls: [], expected: "", included: true, source, agent_version: "" };
}

function caseFromSpan(detail: SpanDetail, source: CaseSource): Candidate {
  const messages = parsedMessages(detail.input);
  const replies = parsedMessages(detail.output).filter((message) => message.role === "assistant");
  const reply = replies.at(-1)?.content ?? detail.output;
  return makeCase(
    messages.length ? messages : [{ role: "user", content: detail.input, name: "", tool_calls: [] }],
    reply,
    source,
  );
}

function textCases(text: string): Candidate[] {
  const lines = text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const parsed = lines.map((line) => {
    try {
      const value: { messages?: unknown; reply?: unknown } = JSON.parse(line);
      return Array.isArray(value.messages) ? value : null;
    } catch {
      return null;
    }
  });
  if (parsed.length === 0 || parsed.some((value) => value === null))
    return [makeCase([{ role: "user", content: text, name: "", tool_calls: [] }], "", NO_SOURCE)];
  return parsed.map((value) =>
    makeCase(
      parsedMessages(JSON.stringify(value?.messages)),
      typeof value?.reply === "string" ? value.reply : "",
      NO_SOURCE,
    ),
  );
}

const isCase = (candidate: Candidate): candidate is DatasetCase => "id" in candidate;

export function demoDatasetsApi(data: LensDemoData, now: () => Date = () => new Date()): DatasetsApi {
  const revisions = new Map<string, Dataset[]>();
  const missing = () => Promise.reject(new ApiError("Dataset not found", 404, { detail: "Dataset not found" }));
  const latest = (id: string) => revisions.get(id)?.at(-1);
  const run = (traceId: string) => data.runs.find(({ trace }) => trace.summary.trace_id === traceId);
  const spanCase = (traceId: string, spanId: string, source: CaseSource): Candidate => {
    const found = run(traceId);
    const detail = spanId
      ? found?.details.find((span) => span.span_id === spanId)
      : found?.details.findLast((span) => found.trace.spans.find((s) => s.span_id === span.span_id)?.type === "llm");
    return detail ? caseFromSpan(detail, source) : { source, reason: "no_content" };
  };
  const sourceCases = (source: BuildSource): Candidate[] => {
    if (source.kind === "text") return textCases(source.text);
    if (source.kind === "finding") {
      const findings = data.lenses
        .flatMap((lens) => lens.jobs.flatMap((job) => job.findings ?? []))
        .filter((finding) => source.finding_ids.includes(finding.id));
      const spans = new Map(
        findings.flatMap((finding) =>
          finding.evidence.map((evidence) => {
            const traceId = evidenceTarget(evidence.execution_id)?.id ?? "";
            const origin = { ...NO_SOURCE, trace_id: traceId, span_id: evidence.span_id, finding_id: finding.id };
            return [`${traceId}/${evidence.span_id}`, { ...origin, lens_id: source.lens_id }] as const;
          }),
        ),
      );
      return [...spans.values()].map((origin) => spanCase(origin.trace_id, origin.span_id, origin));
    }
    const origin = {
      ...NO_SOURCE,
      trace_id: source.trace_id,
      trace_ref: source.trace_ref ?? "",
      span_id: source.span_id ?? "",
    };
    return [spanCase(source.trace_id, source.span_id ?? "", origin)];
  };
  return {
    list: async () =>
      [...revisions.values()].map((history) => {
        const dataset = history.at(-1)!;
        return {
          id: dataset.id,
          name: dataset.name,
          agent_name: dataset.agent_name,
          revision: dataset.revision,
          case_count: dataset.cases.length,
          updated_at: dataset.created_at,
        };
      }),
    get: (id, revision) => {
      const history = revisions.get(id);
      const found = revision === undefined ? history?.at(-1) : history?.find((item) => item.revision === revision);
      return found ? Promise.resolve(found) : missing();
    },
    create: async ({ name, agent_name }) => {
      const dataset: Dataset = {
        id: `demo-dataset-${revisions.size + 1}`,
        name,
        agent_name: agent_name ?? "",
        team_id: "",
        created_at: now().toISOString(),
        revision: 0,
        created_by: "demo",
        cases: [],
      };
      revisions.set(dataset.id, [dataset]);
      return dataset;
    },
    build: async ({ sources, dataset_id }) => {
      const existing = (dataset_id && latest(dataset_id)?.cases) || [];
      const seen = new Set(existing.map((item) => item.id));
      const candidates = sources.flatMap(sourceCases);
      const cases: DatasetCase[] = [];
      const skipped: SkippedCase[] = [];
      for (const candidate of candidates) {
        if (!isCase(candidate)) skipped.push(candidate);
        else if (seen.has(candidate.id)) skipped.push({ source: candidate.source, reason: "duplicate" });
        else if (existing.length + cases.length >= MAX_CASES)
          skipped.push({ source: candidate.source, reason: "over_limit" });
        else {
          seen.add(candidate.id);
          cases.push(candidate);
        }
      }
      return { cases, skipped };
    },
    saveRevision: async (id, { base_revision, cases }) => {
      const current = latest(id);
      if (!current) return missing();
      if (current.revision !== base_revision)
        throw new ApiError("Dataset changed, reload", 409, { detail: "Dataset changed, reload" });
      const saved: Dataset = {
        ...current,
        revision: current.revision + 1,
        created_at: now().toISOString(),
        cases: cases.map((item) => ({
          ...item,
          reply: item.reply ?? "",
          tool_calls: item.tool_calls ?? [],
          expected: item.expected ?? "",
          included: item.included ?? true,
          agent_version: item.agent_version ?? "",
          messages: item.messages.map((message) => ({
            ...message,
            name: message.name ?? "",
            tool_calls: message.tool_calls ?? [],
          })),
          source: { ...NO_SOURCE, ...item.source },
        })),
      };
      revisions.get(id)!.push(saved);
      return saved;
    },
    exportJsonl: async (id, revision) => {
      const history = revisions.get(id);
      const found = revision === undefined ? history?.at(-1) : history?.find((item) => item.revision === revision);
      if (!found) return missing();
      const lines = found.cases.filter((item) => item.included).map((item) => `${JSON.stringify(item)}\n`);
      return new Blob(lines, { type: "application/x-ndjson" });
    },
    evalCases: async (id, revision) => {
      const found = revisions.get(id)?.find((item) => item.revision === revision);
      if (!found) return missing();
      return { dataset_id: id, revision, cases: found.cases.filter((item) => item.included) };
    },
  };
}
