import type { AutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import { isAutoRouterDeployment } from "@/app/(dashboard)/hooks/models/useModels";
import type { ModelGroup } from "@/components/llm_calls/fetch_models";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

export const ROUTER_FAMILIES = {
  claude: "Claude",
  openai: "OpenAI",
  gemini: "Gemini",
  grok: "Grok",
  glm: "GLM",
  deepseek: "DeepSeek",
} as const;
export const ROUTER_GOALS = { cheapest: "Cheapest", balanced: "Balanced", quality: "Best quality" } as const;
export type RouterFamily = keyof typeof ROUTER_FAMILIES;
export type RouterGoal = keyof typeof ROUTER_GOALS;
export type FamilyPreference = RouterFamily | "auto";

type ModelIdentity = {
  family: RouterFamily;
  line: string;
  rank: number;
  version: readonly number[];
  snapshot: number;
  model: string;
};
export type SolverModel = Partial<ModelIdentity> & { name: string; provider: string; model: string };
export type RouterModel = ModelIdentity & SolverModel;
export type SolverPair = { everyday: RouterModel; hard: RouterModel; notice: string | null };

const releaseVersion = (raw: string, date?: string) => {
  const snapshot = raw.match(/[-.](\d{8})$/);
  const release = snapshot ? raw.slice(0, -9) : raw;
  return {
    version: release.split(/[.-]/).map(Number),
    // An undated alias follows the current snapshot of its release.
    snapshot: date || snapshot ? Number((date ?? snapshot![1]).replaceAll("-", "")) : Number.MAX_SAFE_INTEGER,
  };
};

const parseModel = (raw: string): ModelIdentity | null => {
  const model = raw
    .toLowerCase()
    .split("/")
    .at(-1)!
    .replace(/^(?:us|eu|apac|global|au|jp|us-gov)\./, "")
    .replace(/^(?:anthropic|openai|zai|deepseek)\./, "")
    .replace(/^databricks-/, "")
    .replace(/@default$/, "")
    .replace(/-v\d+(?::\d+)?$/, "");
  const versionFirst = model.match(/^claude-(\d+(?:[.-]\d+)*)-(haiku|sonnet|opus)(?:[-@]([0-9]{8}))?$/);
  const claude =
    model.match(/^claude-(haiku|sonnet|opus)-(\d+(?:[.-]\d+)*)(?:@([0-9]+))?$/) ??
    (versionFirst ? [versionFirst[0], versionFirst[2], versionFirst[1], versionFirst[3]] : null);
  if (claude)
    return {
      family: "claude",
      line: claude[1],
      rank: ["haiku", "sonnet", "opus"].indexOf(claude[1]),
      ...releaseVersion(claude[2], claude[3]),
      model,
    };
  const openai = model.match(/^gpt-(\d+(?:[.-]\d+)*)-(luna|terra|sol|astra)(?:-(\d{4}-\d{2}-\d{2}))?$/);
  if (openai)
    return {
      family: "openai",
      line: openai[2],
      rank: openai[2] === "terra" ? 0.5 : ["luna", "sol", "astra"].indexOf(openai[2]),
      ...releaseVersion(openai[1], openai[3]),
      model,
    };
  const gemini = model.match(/^gemini-(\d+(?:[.-]\d+)*)-(flash-lite|flash|pro)(?:-preview)?$/);
  if (gemini)
    return {
      family: "gemini",
      line: gemini[2],
      rank: ["flash-lite", "flash", "pro"].indexOf(gemini[2]),
      ...releaseVersion(gemini[1]),
      model,
    };
  const grok = model.match(/^grok-(\d+(?:[.-]\d+)*?)(?:-(\d{4}))?$/);
  if (grok) return { family: "grok", line: "grok", rank: 2, ...releaseVersion(grok[1], grok[2]), model };
  const glm = model.match(/^glm-(\d+(?:[.-]\d+)*)(-flash)?$/);
  if (glm)
    return {
      family: "glm",
      line: glm[2] ? "flash" : "glm",
      rank: glm[2] ? 0 : 2,
      ...releaseVersion(glm[1]),
      model,
    };
  const deepseek = model.match(/^deepseek-v(\d+(?:[.-]\d+)*)-(flash|pro)$/);
  if (deepseek)
    return {
      family: "deepseek",
      line: deepseek[2],
      rank: deepseek[2] === "flash" ? 0 : 2,
      ...releaseVersion(deepseek[1]),
      model,
    };
  return null;
};

const identityOf = (deployment: AutoRouterDeployment): ModelIdentity | null =>
  parseModel(deployment.litellm_params?.model ?? "") ??
  parseModel(deployment.model_info?.base_model ?? deployment.litellm_params?.base_model ?? "");

const providerOf = (deployment: AutoRouterDeployment, group: ModelGroup, family?: RouterFamily): string => {
  const raw = deployment.litellm_params?.model ?? "";
  const provider =
    deployment.litellm_params?.custom_llm_provider ||
    (raw.includes("/") ? raw.split("/")[0] : group.providers?.join(", ")) ||
    (family
      ? { claude: "anthropic", glm: "zai", openai: "openai", gemini: "gemini", grok: "xai", deepseek: "deepseek" }[
          family
        ]
      : "");
  if (provider.startsWith("bedrock")) return "Bedrock";
  return (
    (
      {
        anthropic: "Anthropic",
        openai: "OpenAI",
        gemini: "Google AI",
        xai: "xAI",
        azure: "Azure",
        azure_ai: "Azure AI",
        databricks: "Databricks",
        github_copilot: "GitHub Copilot",
        "vertex_ai-anthropic_models": "Vertex AI",
        vertex_ai: "Vertex AI",
        zai: "Z.ai",
        deepseek: "DeepSeek",
      } as Record<string, string>
    )[provider] ?? provider
  );
};

const compareVersion = (left: readonly number[], right: readonly number[]): number => {
  const index = Array.from({ length: Math.max(left.length, right.length) }, (_, i) => i).find(
    (i) => (left[i] ?? 0) !== (right[i] ?? 0),
  );
  return index === undefined ? 0 : (left[index] ?? 0) - (right[index] ?? 0);
};

const compareRelease = (left: ModelIdentity, right: ModelIdentity): number =>
  compareVersion(left.version, right.version) || left.snapshot - right.snapshot;

export const deployedRouterModels = (
  groups: readonly ModelGroup[],
  deployments: readonly AutoRouterDeployment[],
): SolverModel[] =>
  groups
    .flatMap((group) => {
      if (group.mode && group.mode !== "chat") return [];
      const rows = deployments.filter((row) => row.model_name === group.model_group);
      if (
        rows.some(isAutoRouterDeployment) ||
        group.model_group.includes("*") ||
        group.model_group.startsWith("auto_router/")
      )
        return [];
      const active = rows.filter((row) => !row.model_info?.blocked);
      if (rows.length > 0 && active.length === 0) return [];
      if (active.some((row) => row.model_info?.mode && row.model_info.mode !== "chat")) return [];
      if (active.some((row) => row.litellm_params?.model?.includes("*"))) return [];
      const identities = active.length > 0 ? active.map(identityOf) : [parseModel(group.model_group)];
      const first = identities[0];
      const homogeneous =
        first &&
        identities.every((identity) => {
          if (!identity) return false;
          const sameLine = identity.family === first.family && identity.line === first.line;
          return sameLine && compareRelease(identity, first) === 0;
        });
      const identity = homogeneous ? first : undefined;
      const providers =
        active.length > 0
          ? active.map((row) => providerOf(row, group, identity?.family))
          : [providerOf({}, group, identity?.family)];
      return [
        {
          ...identity,
          name: group.model_group,
          model: identity?.model ?? group.model_group,
          provider: [...new Set(providers)].toSorted().join(", "),
        },
      ];
    })
    .toSorted((a, b) => a.name.localeCompare(b.name));

const latestLines = (models: readonly RouterModel[]): RouterModel[] =>
  models.filter(
    (model) =>
      !models.some(
        (other) => other.family === model.family && other.line === model.line && compareRelease(other, model) > 0,
      ),
  );

const requestedRanks = (goal: RouterGoal): readonly [number, number] =>
  (({ cheapest: [0, 1], quality: [1, 2], balanced: [0, 2] }) as const)[goal];

export const recommendSolverPair = (
  models: readonly SolverModel[],
  goal: RouterGoal,
  family: FamilyPreference,
): SolverPair | null => {
  const recognized = models.filter((model): model is RouterModel => model.family !== undefined);
  const latest = latestLines(recognized);
  const [low, high] = requestedRanks(goal);
  const pairs = latest.flatMap((everyday) =>
    latest
      .filter((hard) => {
        const matchesFamily = family === "auto" || everyday.family === family || hard.family === family;
        const ordered = hard.rank > everyday.rank;
        const equalRankFallback = hard.rank === everyday.rank && hard.family !== everyday.family;
        return (ordered || equalRankFallback) && matchesFamily;
      })
      .map((hard) => ({ everyday, hard })),
  );
  const score = ({ everyday, hard }: { everyday: RouterModel; hard: RouterModel }): number =>
    (everyday.family === hard.family ? 0 : 100) +
    (everyday.rank === hard.rank ? 10 : 0) +
    Math.abs(everyday.rank - low) +
    Math.abs(hard.rank - high);
  const chosen = pairs.toSorted(
    (a, b) =>
      score(a) - score(b) || a.everyday.name.localeCompare(b.everyday.name) || a.hard.name.localeCompare(b.hard.name),
  )[0];
  if (!chosen) return null;
  const mixed = chosen.everyday.family !== chosen.hard.family;
  const substituted = chosen.everyday.rank !== low || chosen.hard.rank !== high;
  const mixedNotice = mixed ? "A complete pair is unavailable in one family. Using two model families." : null;
  const substituteNotice = substituted ? "Using the closest available pair in this family." : null;
  return { ...chosen, notice: mixedNotice ?? substituteNotice };
};

export const solverPairConfig = (everyday: string, hard: string): ComplexityRouterConfigValue => ({
  classifier_type: "heuristic",
  classification_mode: "user_turn",
  tiers: { SIMPLE: [everyday], MEDIUM: [everyday], COMPLEX: [hard], REASONING: [hard] },
});

export const simpleSetupError = (
  loading: boolean,
  failed: boolean,
  everyday?: SolverModel,
  hard?: SolverModel,
): string | null => {
  if (loading) return "Loading deployed models";
  if (failed) return "Could not verify deployed models";
  if (!everyday || !hard) return "Select two different deployed models";
  if (everyday.name === hard.name) return "Select two different deployed models";
  return null;
};

export const nextRouterName = (names: readonly string[]): string => {
  const candidates = [
    "auto-router",
    ...Array.from({ length: names.length + 1 }, (_, index) => `auto-router-${index + 2}`),
  ];
  return candidates.find((name) => !names.includes(name))!;
};
