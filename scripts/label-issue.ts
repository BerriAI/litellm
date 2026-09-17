#!/usr/bin/env bun

import { githubApi, listAll, type Comment, type GitHubApi } from "./auto-close-duplicates";
import type { GateVerdict, Verdict } from "./classify-issue";
import { MANIFEST, NAMESPACES, labelName, manifestLabels, namespaceOf, type Namespace } from "./issue-labels";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface LabelConfig {
  readonly repo: string;
  readonly issueNumber: number;
  readonly dryRun: boolean;
}

export interface LabelPlan {
  readonly add: readonly string[];
  readonly remove: readonly string[];
}

export interface LabelOutcome {
  readonly plan: LabelPlan;
  readonly comment: string | null;
  readonly removedNotices: number;
}

export type ParsedVerdict =
  | { readonly kind: "verdict"; readonly verdict: Verdict }
  | { readonly kind: "invalid"; readonly reason: string };

export const TEMPLATE_MARKER = "<!-- litellm:needs-template -->";
export const BOT_LOGIN = "github-actions[bot]";
const TEMPLATE_URLS: Readonly<Record<GateVerdict["template"], string>> = {
  bug: "https://github.com/BerriAI/litellm/issues/new?template=bug_report.yml",
  feature: "https://github.com/BerriAI/litellm/issues/new?template=feature_request.yml",
};

export function desiredLabels(verdict: Verdict): readonly string[] {
  if (verdict.gate === "template") {
    return [labelName("needs", "template")];
  }
  return [
    labelName("domain", verdict.domain),
    ...(verdict.provider === null ? [] : [labelName("provider", verdict.provider)]),
    labelName("kind", verdict.kind),
    labelName("priority", verdict.priority),
    labelName("lift", verdict.lift),
    ...verdict.needs.map((need) => labelName("needs", need)),
  ];
}

function touchedNamespaces(verdict: Verdict): readonly Namespace[] {
  return verdict.gate === "template" ? ["needs"] : NAMESPACES;
}

export function labelPlan(current: readonly string[], verdict: Verdict): LabelPlan {
  const desired = desiredLabels(verdict);
  const touched = touchedNamespaces(verdict);
  const remove = current.filter((label) => {
    const namespace = namespaceOf(label);
    return namespace !== undefined && touched.includes(namespace) && !desired.includes(label);
  });
  const add = desired.filter((label) => !current.includes(label));
  return { add, remove };
}

export function templateComment(verdict: GateVerdict): string {
  const named = verdict.missing.map((heading) => `**${heading}**`).join(", ");
  const pronoun = verdict.missing.length === 1 ? "it" : "them";
  return [
    TEMPLATE_MARKER,
    `This issue is missing ${named} from the [${verdict.template} template](${TEMPLATE_URLS[verdict.template]}). Edit the description to add ${pronoun} and it will be labelled automatically.`,
  ].join("\n");
}

export function parseVerdict(raw: string): ParsedVerdict {
  const parsed = ((): unknown => {
    try {
      return JSON.parse(raw);
    } catch {
      return undefined;
    }
  })();
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { kind: "invalid", reason: "the verdict is not a JSON object" };
  }
  const verdict = parsed as Verdict;
  if (verdict.gate === "template") {
    const missing = Array.isArray(verdict.missing) ? verdict.missing.filter((item) => typeof item === "string") : [];
    if (missing.length === 0 || (verdict.template !== "bug" && verdict.template !== "feature")) {
      return { kind: "invalid", reason: "a template verdict needs a template and at least one missing section" };
    }
    return { kind: "verdict", verdict: { gate: "template", template: verdict.template, missing } };
  }
  if (verdict.gate !== "pass" || !Array.isArray(verdict.needs)) {
    return { kind: "invalid", reason: `gate must be "pass" or "template", got ${JSON.stringify(verdict.gate)}` };
  }
  const known = new Set(manifestLabels(MANIFEST).map((label) => label.name));
  const unknown = desiredLabels(verdict).filter((label) => !known.has(label));
  if (unknown.length > 0) {
    return { kind: "invalid", reason: `not in .github/labels.json: ${unknown.join(", ")}` };
  }
  return { kind: "verdict", verdict };
}

export async function labelIssue(api: GitHubApi, config: LabelConfig, verdict: Verdict): Promise<LabelOutcome> {
  const issuePath = `/repos/${config.repo}/issues/${config.issueNumber}`;
  const issue = await api.request<{ readonly labels: readonly { readonly name: string }[] }>("GET", issuePath);
  const plan = labelPlan(
    issue.labels.map((label) => label.name),
    verdict,
  );
  const comments = await listAll<Comment>(api, `${issuePath}/comments`);
  const notices = comments.filter((comment) => comment.user.login === BOT_LOGIN && comment.body.includes(TEMPLATE_MARKER));
  const comment = verdict.gate === "template" && notices.length === 0 ? templateComment(verdict) : null;
  const staleNotices = verdict.gate === "pass" ? notices : [];
  if (config.dryRun) {
    return { plan, comment, removedNotices: staleNotices.length };
  }
  for (const label of plan.remove) {
    await api.request("DELETE", `${issuePath}/labels/${encodeURIComponent(label)}`);
  }
  if (plan.add.length > 0) {
    await api.request("POST", `${issuePath}/labels`, { labels: plan.add });
  }
  if (comment !== null) {
    await api.request("POST", `${issuePath}/comments`, { body: comment });
  }
  for (const notice of staleNotices) {
    await api.request("DELETE", `/repos/${config.repo}/issues/comments/${notice.id}`);
  }
  return { plan, comment, removedNotices: staleNotices.length };
}

export function readConfig(env: Readonly<Record<string, string | undefined>>): LabelConfig & { readonly token: string } {
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

function describe(config: LabelConfig, outcome: LabelOutcome): string {
  const changes = [
    ...outcome.plan.add.map((label) => `+${label}`),
    ...outcome.plan.remove.map((label) => `-${label}`),
    ...(outcome.removedNotices > 0 ? [`-${outcome.removedNotices} needs-template comment(s)`] : []),
  ];
  const summary = changes.length === 0 ? "nothing to change" : changes.join(" ");
  const commentNote = outcome.comment === null ? "" : `\n\n${outcome.comment}`;
  if (config.dryRun) {
    return `#${config.issueNumber}: DRY RUN, set the ISSUE_CLASSIFIER_ENABLED repo variable to true to apply: ${summary}${commentNote}`;
  }
  return `#${config.issueNumber}: ${summary}${outcome.comment === null ? "" : ", commented"}`;
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env);
  const parsed = parseVerdict(process.env.VERDICT ?? "");
  if (parsed.kind === "invalid") {
    throw new Error(`refusing to label #${config.issueNumber}: ${parsed.reason}`);
  }
  const outcome = await labelIssue(githubApi(token), config, parsed.verdict);
  console.log(describe(config, outcome));
}
