import { parseStartNumbers, type ShadowEvalValues } from "./schema";

export const buildStartBody = (inputs: ShadowEvalValues, judgeModel: string) => {
  const { parsedPct, parsedMaxBudget } = parseStartNumbers(inputs);
  return {
    api_key_ids: inputs.apiKeyIds,
    team_ids: inputs.teamIds,
    user_ids: inputs.userIds,
    models: inputs.direction === "forward" ? inputs.models : [],
    router_names: inputs.routerNames,
    direction: inputs.direction,
    ...(inputs.direction === "reverse" ? { baseline_model: inputs.baselineModel ?? undefined } : {}),
    shadow_percentage: parsedPct,
    duration_days: Number.parseInt(inputs.durationDays, 10),
    max_budget: parsedMaxBudget,
    judge_model: judgeModel,
    skip_same_model: false,
  };
};
