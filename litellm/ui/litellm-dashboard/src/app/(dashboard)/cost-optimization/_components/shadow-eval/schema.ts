import { z } from "zod";

export const MAX_ROUTERS = 4;
export const MAX_MODELS = 100;

const shadowEvalShape = {
  apiKeyIds: z.array(z.string()),
  teamIds: z.array(z.string()),
  userIds: z.array(z.string()),
  models: z.array(z.string()),
  routerNames: z.array(z.string()),
  direction: z.enum(["forward", "reverse"]),
  baselineModel: z.string().nullable(),
  percentage: z.string(),
  durationDays: z.string(),
  judgeModel: z.string().nullable(),
  maxBudget: z.string(),
};
const shadowEvalFields = z.object(shadowEvalShape);

export type ShadowEvalValues = z.infer<typeof shadowEvalFields>;
export type ShadowEvalDirection = ShadowEvalValues["direction"];

export const shadowEvalDefaults: ShadowEvalValues = {
  apiKeyIds: [],
  teamIds: [],
  userIds: [],
  models: [],
  routerNames: [],
  direction: "forward",
  baselineModel: null,
  percentage: "10",
  durationDays: "7",
  judgeModel: null,
  maxBudget: "10",
};

export function parseStartNumbers(inputs: Pick<ShadowEvalValues, "percentage" | "maxBudget">) {
  const parsedPct = Number.parseFloat(inputs.percentage);
  const percentageValid = parsedPct >= 0.1 && parsedPct <= 100;
  const parsedMaxBudget = Number.parseFloat(inputs.maxBudget);
  const maxBudgetValid = parsedMaxBudget >= 0.01 && parsedMaxBudget <= 10000;
  return { parsedPct, parsedMaxBudget, percentageValid, maxBudgetValid };
}

export const shadowEvalSchema = shadowEvalFields.refine((inputs) => {
  const { percentageValid, maxBudgetValid } = parseStartNumbers(inputs);
  const baselinePicked = inputs.direction === "forward" || Boolean(inputs.baselineModel);
  const targetsPicked = inputs.apiKeyIds.length + inputs.teamIds.length + inputs.userIds.length > 0;
  const routerCountValid = inputs.routerNames.length >= 1 && inputs.routerNames.length <= MAX_ROUTERS;
  const routersMatchDirection = inputs.direction === "forward" || inputs.routerNames.length === 1;
  const routersValid = routerCountValid && routersMatchDirection;
  const scopeValid = routersValid && (inputs.direction === "reverse" || inputs.models.length <= MAX_MODELS);
  const modelsPicked = scopeValid && Boolean(inputs.judgeModel) && baselinePicked;
  const filled = targetsPicked && modelsPicked;
  const boundsValid = percentageValid && maxBudgetValid;
  return filled && boundsValid;
});
