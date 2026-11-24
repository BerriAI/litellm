#!/usr/bin/env bun

import {
  DEFAULT_GRACE_DAYS,
  FLAG_LABEL,
  duplicateTarget,
  githubApi,
  listAll,
  type Comment,
  type GitHubApi,
  type Issue,
} from "./auto-close-duplicates";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface Verdict {
  readonly duplicate_of: number | null;
  readonly confidence: number;
  readonly evidence: string;
}

export interface FlagConfig {
  readonly repo: string;
  readonly issueNumber: number;
  readonly dryRun: boolean;
}

export type ParsedVerdict =
  | { readonly kind: "verdict"; readonly verdict: Verdict }
  | { readonly kind: "skip"; readonly reason: string };

export type FlagTarget =
  | { readonly kind: "target"; readonly original: number }
  | { readonly kind: "skip"; readonly reason: string };

export type FlagVerdict =
  | { readonly kind: "flagged"; readonly original: number; readonly body: string }
  | { readonly kind: "skip"; readonly reason: string };

export const MIN_CONFIDENCE = 0.95;
export const NOTICE_MARKER_PREFIX = "<!-- litellm:potential-duplicate candidates=";

const skip = (reason: string): { readonly kind: "skip"; readonly reason: string } => ({ kind: "skip", reason });

const parseJson = (raw: string): unknown => {
  try {
    return JSON.parse(raw);
  } catch {
    return undefined;
  }
};

export function parseVerdict(raw: string): ParsedVerdict {
  const parsed = parseJson(raw);
  if (typeof parsed !== "object" || parsed === null) {
    return skip("Codex did not return a JSON object");
  }
  const { duplicate_of, confidence, evidence } = parsed as Record<string, unknown>;
  if (duplicate_of !== null && !Number.isInteger(duplicate_of)) {
    return skip(`duplicate_of must be an integer or null, got ${JSON.stringify(duplicate_of)}`);
  }
  if (typeof confidence !== "number" || !Number.isFinite(confidence)) {
    return skip(`confidence must be a number, got ${JSON.stringify(confidence)}`);
  }
  if (typeof evidence !== "string" || evidence.trim() === "") {
    return skip("evidence must be a non-empty string");
  }
  return { kind: "verdict", verdict: { duplicate_of: duplicate_of as number | null, confidence, evidence } };
}

export function flagTarget(verdict: Verdict, issueNumber: number): FlagTarget {
  if (verdict.duplicate_of === null) {
    return skip("no duplicate named");
  }
  if (verdict.confidence < MIN_CONFIDENCE) {
    return skip(`confidence ${verdict.confidence} is below ${MIN_CONFIDENCE}`);
  }
  if (verdict.duplicate_of >= issueNumber) {
    return skip(`#${verdict.duplicate_of} is not older than #${issueNumber}`);
  }
  return { kind: "target", original: verdict.duplicate_of };
}

export function noticeBody(issue: Issue, prior: Issue, evidence: string): string {
  const closed = prior.state === "closed";
  const lead = closed
    ? `**Already reported in #${prior.number}**, which is closed`
    : `**Possible duplicate of #${prior.number}**`;
  const ask = closed
    ? "If that issue covers this one, follow up there. If this is a new case, say so here and a maintainer will take the label off."
    : `If that is right, add a thumbs-up to #${prior.number} and follow along there. If it is not, say so here and a maintainer will take the label off.`;
  const autoCloses = duplicateTarget(issue, [prior], []).kind === "close";
  const warning = autoCloses
    ? `\n\nYour title is identical to #${prior.number}, so this issue closes automatically in ${DEFAULT_GRACE_DAYS} days unless someone responds here.`
    : "";
  return [`${NOTICE_MARKER_PREFIX}${prior.number}, -->`, lead, "", evidence, "", ask + warning].join("\n");
}

export async function flagIssue(api: GitHubApi, config: FlagConfig, verdict: Verdict): Promise<FlagVerdict> {
  const target = flagTarget(verdict, config.issueNumber);
  if (target.kind === "skip") {
    return target;
  }
  const issuePath = `/repos/${config.repo}/issues/${config.issueNumber}`;
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  if (comments.some((comment) => comment.body.includes(NOTICE_MARKER_PREFIX))) {
    return skip("already carries a duplicate notice");
  }
  const prior = await api.request<Issue>("GET", `/repos/${config.repo}/issues/${target.original}`);
  if (prior.pull_request !== undefined) {
    return skip(`#${target.original} is a pull request`);
  }
  const issue = await api.request<Issue>("GET", issuePath);
  const body = noticeBody(issue, prior, verdict.evidence);
  if (!config.dryRun) {
    await api.request("POST", `${issuePath}/labels`, { labels: [FLAG_LABEL] });
    await api.request("POST", `${issuePath}/comments`, { body });
  }
  return { kind: "flagged", original: target.original, body };
}

export function readConfig(env: Readonly<Record<string, string | undefined>>): FlagConfig & { readonly token: string } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/repo) are required");
  }
  const issueNumber = Number(env.ISSUE_NUMBER);
  if (!Number.isInteger(issueNumber) || issueNumber <= 0) {
    throw new Error(`ISSUE_NUMBER must be a positive integer, got "${env.ISSUE_NUMBER}"`);
  }
  return { token, repo, issueNumber, dryRun: env.DRY_RUN === "true" };
}

function describe(config: FlagConfig, verdict: FlagVerdict): string {
  if (verdict.kind === "skip") {
    return `#${config.issueNumber}: skipped, ${verdict.reason}`;
  }
  if (config.dryRun) {
    return `#${config.issueNumber}: DRY RUN, set the DUPLICATE_CHECK_ENABLED repo variable to true to post this:\n\n${verdict.body}`;
  }
  return `#${config.issueNumber}: flagged as a possible duplicate of #${verdict.original}`;
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env);
  const parsed = parseVerdict(process.env.VERDICT ?? "");
  const verdict = parsed.kind === "skip" ? parsed : await flagIssue(githubApi(token), config, parsed.verdict);
  console.log(describe(config, verdict));
}
