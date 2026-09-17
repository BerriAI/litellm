#!/usr/bin/env bun

import { githubApi, type GitHubApi } from "./auto-close-duplicates";
import { MANIFEST, type Manifest } from "./issue-labels";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };
declare const Bun: {
  readonly file: (path: string) => { readonly text: () => Promise<string>; readonly json: () => Promise<unknown> };
};

export interface IssueForClassification {
  readonly number: number;
  readonly title: string;
  readonly body: string | null;
  readonly author_association: string;
  readonly pull_request?: unknown;
}

export type Template = "bug" | "feature";

export type Gate =
  | {
      readonly kind: "pass";
      readonly template: Template;
      readonly domainHint: string | null;
      readonly version: string | null;
    }
  | { readonly kind: "template"; readonly template: Template; readonly missing: readonly string[] };

export interface Classification {
  readonly gate: "pass";
  readonly domain: string;
  readonly provider: string | null;
  readonly kind: string;
  readonly priority: string;
  readonly lift: string;
  readonly route: string | null;
  readonly version: string | null;
  readonly needs: readonly string[];
  readonly reason: string;
}

export interface GateVerdict {
  readonly gate: "template";
  readonly template: Template;
  readonly missing: readonly string[];
}

export type Verdict = Classification | GateVerdict;

export type ParsedClassification =
  | { readonly kind: "classification"; readonly classification: Classification }
  | { readonly kind: "invalid"; readonly reason: string };

export interface ChatMessage {
  readonly role: "system" | "user";
  readonly content: string;
}

export interface ChatRequest {
  readonly model: string;
  readonly messages: readonly ChatMessage[];
  readonly response_format: {
    readonly type: "json_schema";
    readonly json_schema: { readonly name: string; readonly strict: true; readonly schema: object };
  };
}

export interface LlmClient {
  readonly complete: (request: ChatRequest) => Promise<string>;
}

export interface ClassifyConfig {
  readonly repo: string;
  readonly issueNumber: number;
  readonly model: string;
}

export interface Schema {
  readonly properties: Readonly<Record<string, { readonly enum?: readonly (string | null)[] }>>;
}

export const BUG_SECTIONS = ["Description", "Config", "LiteLLM Version", "Steps to Repro"] as const;
export const FEATURE_SECTIONS = ["The Feature", "User Flow", "How far you got"] as const;
export const DOMAIN_HEADING = "Which part of LiteLLM is this about?";
export const VERSION_HEADING = "LiteLLM Version";
export const MIN_SECTION_CHARS = 20;
export const BODY_CAP_CHARS = 8000;
export const MAINTAINER_ASSOCIATIONS: readonly string[] = ["OWNER", "MEMBER", "COLLABORATOR"];
const EMPTY_FIELD = "_No response_";
const NOT_SURE = "Not sure";

export function sections(body: string): ReadonlyMap<string, string> {
  const parts = body.split(/^### (.+)$/m).slice(1);
  const pairs = parts.flatMap((part, index): readonly (readonly [string, string])[] =>
    index % 2 === 0 ? [[part.trim(), (parts[index + 1] ?? "").trim()]] : [],
  );
  return new Map(pairs);
}

export function templateFor(title: string, found: ReadonlyMap<string, string>): Template {
  if (/^\s*\[bug\]/i.test(title)) {
    return "bug";
  }
  if (/^\s*\[feature\]/i.test(title)) {
    return "feature";
  }
  return FEATURE_SECTIONS.some((heading) => found.has(heading)) ? "feature" : "bug";
}

function hasSubstance(heading: string, text: string | undefined): boolean {
  if (text === undefined || text === "" || text === EMPTY_FIELD) {
    return false;
  }
  if (heading === VERSION_HEADING) {
    return /\d+\.\d+/.test(text);
  }
  return text.length >= MIN_SECTION_CHARS;
}

export function gate(issue: Pick<IssueForClassification, "title" | "body" | "author_association">): Gate {
  const found = sections(issue.body ?? "");
  const template = templateFor(issue.title, found);
  const required: readonly string[] = template === "bug" ? BUG_SECTIONS : FEATURE_SECTIONS;
  const missing = required.filter((heading) => !hasSubstance(heading, found.get(heading)));
  if (missing.length > 0 && !MAINTAINER_ASSOCIATIONS.includes(issue.author_association)) {
    return { kind: "template", template, missing };
  }
  const hint = found.get(DOMAIN_HEADING);
  const version = found.get(VERSION_HEADING);
  return {
    kind: "pass",
    template,
    domainHint: hint === undefined || hint === EMPTY_FIELD || hint === NOT_SURE ? null : hint,
    version: hasSubstance(VERSION_HEADING, version) ? (version ?? null) : null,
  };
}

export function userMessage(issue: Pick<IssueForClassification, "title" | "body">, passed: Gate & { kind: "pass" }): string {
  const body = issue.body ?? "";
  const capped =
    body.length > BODY_CAP_CHARS
      ? `${body.slice(0, BODY_CAP_CHARS)}\n\n[body truncated at ${BODY_CAP_CHARS} characters]`
      : body;
  const versionLine = passed.version === null ? "" : `\nLiteLLM Version (from the template): ${passed.version}`;
  return [
    `Title: ${issue.title}`,
    `Template: ${passed.template}`,
    `Reporter's pick from the domain dropdown: ${passed.domainHint ?? "none"}${versionLine}`,
    "",
    capped,
  ].join("\n");
}

export function buildRequest(
  model: string,
  prompt: string,
  schema: object,
  issue: Pick<IssueForClassification, "title" | "body">,
  passed: Gate & { kind: "pass" },
): ChatRequest {
  return {
    model,
    messages: [
      { role: "system", content: prompt },
      { role: "user", content: userMessage(issue, passed) },
    ],
    response_format: { type: "json_schema", json_schema: { name: "issue_classification", strict: true, schema } },
  };
}

export function routesOf(schema: Schema): readonly string[] {
  return (schema.properties.route?.enum ?? []).filter((value): value is string => typeof value === "string");
}

const invalid = (reason: string): ParsedClassification => ({ kind: "invalid", reason });

const parseJson = (raw: string): unknown => {
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
};

function enumValue(
  fields: Readonly<Record<string, unknown>>,
  field: string,
  allowed: readonly string[],
): { readonly ok: true; readonly value: string } | { readonly ok: false; readonly reason: string } {
  const value = fields[field];
  if (typeof value !== "string" || !allowed.includes(value)) {
    return { ok: false, reason: `${field} must be one of ${allowed.join(", ")}, got ${JSON.stringify(value)}` };
  }
  return { ok: true, value };
}

export function parseClassification(raw: string, manifest: Manifest, routes: readonly string[]): ParsedClassification {
  const parsed = parseJson(raw);
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return invalid("the model did not return a JSON object");
  }
  const fields = parsed as Readonly<Record<string, unknown>>;
  const domain = enumValue(fields, "domain", Object.keys(manifest.domain));
  const kind = enumValue(fields, "kind", Object.keys(manifest.kind));
  const priority = enumValue(fields, "priority", Object.keys(manifest.priority));
  const lift = enumValue(fields, "lift", Object.keys(manifest.lift));
  const provider = fields.provider === null ? { ok: true as const, value: null } : enumValue(fields, "provider", Object.keys(manifest.provider));
  const route = fields.route === null ? { ok: true as const, value: null } : enumValue(fields, "route", routes);
  const failed = [domain, kind, priority, lift, provider, route].find((result) => !result.ok);
  if (failed !== undefined && !failed.ok) {
    return invalid(failed.reason);
  }
  if (!domain.ok || !kind.ok || !priority.ok || !lift.ok || !provider.ok || !route.ok) {
    return invalid("unreachable");
  }
  const { version, needs_repro: needsRepro, reason } = fields;
  if (version !== null && (typeof version !== "string" || version.trim() === "")) {
    return invalid(`version must be a non-empty string or null, got ${JSON.stringify(version)}`);
  }
  if (typeof needsRepro !== "boolean") {
    return invalid(`needs_repro must be a boolean, got ${JSON.stringify(needsRepro)}`);
  }
  if (typeof reason !== "string" || reason.trim() === "") {
    return invalid("reason must be a non-empty string");
  }
  const isBug = kind.value === "bug";
  return {
    kind: "classification",
    classification: {
      gate: "pass",
      domain: domain.value,
      provider: provider.value,
      kind: kind.value,
      priority: isBug ? priority.value : "p3",
      lift: lift.value,
      route: route.value,
      version: version as string | null,
      needs: [...(version === null ? ["version"] : []), ...(isBug && needsRepro ? ["repro"] : [])],
      reason,
    },
  };
}

export async function classifyIssue(
  api: GitHubApi,
  llm: LlmClient,
  config: ClassifyConfig,
  prompt: string,
  schema: Schema,
): Promise<Verdict> {
  const issue = await api.request<IssueForClassification>("GET", `/repos/${config.repo}/issues/${config.issueNumber}`);
  if (issue.pull_request !== undefined) {
    throw new Error(`#${config.issueNumber} is a pull request`);
  }
  const passed = gate(issue);
  if (passed.kind === "template") {
    return { gate: "template", template: passed.template, missing: passed.missing };
  }
  const raw = await llm.complete(buildRequest(config.model, prompt, schema, issue, passed));
  const parsed = parseClassification(raw, MANIFEST, routesOf(schema));
  if (parsed.kind === "invalid") {
    throw new Error(`the model's answer failed validation: ${parsed.reason}\n${raw}`);
  }
  return parsed.classification;
}

export function litellmClient(apiBase: string, apiKey: string): LlmClient {
  return {
    complete: async (request: ChatRequest): Promise<string> => {
      const response = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/chat/completions`, {
        method: "POST",
        headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        throw new Error(`chat completion failed: ${response.status} ${response.statusText}`);
      }
      const payload = (await response.json()) as {
        readonly choices?: readonly {
          readonly finish_reason?: string;
          readonly message?: { readonly content?: string | null; readonly refusal?: string | null };
        }[];
      };
      const choice = payload.choices?.[0];
      if (choice?.message?.refusal) {
        throw new Error(`the model refused: ${choice.message.refusal}`);
      }
      if (choice?.finish_reason === "length") {
        throw new Error("the model ran out of output tokens before finishing the JSON");
      }
      const content = choice?.message?.content;
      if (typeof content !== "string" || content === "") {
        throw new Error("the model returned no content");
      }
      return content;
    },
  };
}

export function readConfig(
  env: Readonly<Record<string, string | undefined>>,
): ClassifyConfig & { readonly token: string; readonly apiBase: string; readonly apiKey: string } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/repo) are required");
  }
  const issueNumber = Number(env.ISSUE_NUMBER);
  if (!Number.isInteger(issueNumber) || issueNumber <= 0) {
    throw new Error(`ISSUE_NUMBER must be a positive integer, got "${env.ISSUE_NUMBER}"`);
  }
  const apiBase = env.LITELLM_API_BASE;
  const apiKey = env.LITELLM_API_KEY;
  const model = env.ISSUE_CLASSIFIER_MODEL;
  if (!apiBase || !/^https?:\/\//.test(apiBase)) {
    throw new Error("LITELLM_API_BASE must be the URL of a LiteLLM proxy, e.g. https://llm.example.com");
  }
  if (!apiKey) {
    throw new Error("LITELLM_API_KEY is required");
  }
  if (!model) {
    throw new Error("ISSUE_CLASSIFIER_MODEL must name a model the LiteLLM deployment serves");
  }
  return { token, repo, issueNumber, apiBase, apiKey, model };
}

if (import.meta.main) {
  const { token, apiBase, apiKey, ...config } = readConfig(process.env);
  const prompt = await Bun.file(`${import.meta.dir}/../.github/prompts/issue-classifier.md`).text();
  const schema = (await Bun.file(`${import.meta.dir}/../.github/prompts/issue-classifier.schema.json`).json()) as Schema;
  const verdict = await classifyIssue(githubApi(token), litellmClient(apiBase, apiKey), config, prompt, schema);
  console.log(JSON.stringify(verdict));
}
