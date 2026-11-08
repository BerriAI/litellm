import { parseAsString, parseAsStringLiteral } from "nuqs";
import { ISSUE_FAMILIES, ISSUE_STATUSES } from "./issueApi";

export const issueRouteParsers = {
  alert: parseAsString.withOptions({ history: "push" }),
  family: parseAsStringLiteral(ISSUE_FAMILIES),
  status: parseAsStringLiteral(ISSUE_STATUSES),
  kind: parseAsStringLiteral(["performance", "model_deprecation"] as const),
  model: parseAsString,
};
