#!/usr/bin/env bun

import { githubApi, listAll, type GitHubApi } from "./auto-close-duplicates";
import { MANIFEST, manifestLabels, type Manifest, type ManifestLabel } from "./issue-labels";

declare const process: { readonly env: Readonly<Record<string, string | undefined>> };

export interface SyncConfig {
  readonly repo: string;
  readonly dryRun: boolean;
}

export interface GitHubLabel {
  readonly name: string;
  readonly color: string;
  readonly description: string | null;
}

export interface SyncAction extends ManifestLabel {
  readonly kind: "create" | "update" | "unchanged";
}

export function syncPlan(existing: readonly GitHubLabel[], source: Manifest): readonly SyncAction[] {
  const byName = new Map(existing.map((label) => [label.name.toLowerCase(), label]));
  return manifestLabels(source).map((label) => {
    const current = byName.get(label.name.toLowerCase());
    if (current === undefined) {
      return { kind: "create", ...label };
    }
    const same =
      current.color.toLowerCase() === label.color.toLowerCase() && (current.description ?? "") === label.description;
    return { kind: same ? "unchanged" : "update", ...label };
  });
}

export async function syncLabels(api: GitHubApi, config: SyncConfig, source: Manifest): Promise<readonly SyncAction[]> {
  const existing = await listAll<GitHubLabel>(api, `/repos/${config.repo}/labels`);
  const plan = syncPlan(existing, source);
  if (config.dryRun) {
    return plan;
  }
  for (const action of plan) {
    if (action.kind === "create") {
      await api.request("POST", `/repos/${config.repo}/labels`, {
        name: action.name,
        color: action.color,
        description: action.description,
      });
    }
    if (action.kind === "update") {
      await api.request("PATCH", `/repos/${config.repo}/labels/${encodeURIComponent(action.name)}`, {
        color: action.color,
        description: action.description,
      });
    }
  }
  return plan;
}

export function readConfig(env: Readonly<Record<string, string | undefined>>): SyncConfig & { readonly token: string } {
  const token = env.GITHUB_TOKEN;
  const repo = env.GITHUB_REPOSITORY;
  if (!token || !repo || !/^[\w.-]+\/[\w.-]+$/.test(repo)) {
    throw new Error("GITHUB_TOKEN and GITHUB_REPOSITORY (owner/repo) are required");
  }
  return { token, repo, dryRun: env.DRY_RUN === "true" };
}

if (import.meta.main) {
  const { token, ...config } = readConfig(process.env);
  const plan = await syncLabels(githubApi(token), config, MANIFEST);
  const verb = config.dryRun ? "would" : "did";
  for (const action of plan.filter((item) => item.kind !== "unchanged")) {
    console.log(`${action.kind} ${action.name} (#${action.color}) ${action.description}`);
  }
  const count = (kind: SyncAction["kind"]): number => plan.filter((action) => action.kind === kind).length;
  console.log(
    `${verb} create ${count("create")}, update ${count("update")}, leave ${count("unchanged")} unchanged in ${config.repo}`,
  );
}
