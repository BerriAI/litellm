import { parseAsString, parseAsStringLiteral } from "nuqs";
import { ISSUE_FAMILIES, ISSUE_STATUSES } from "./issueApi";

export const issueRouteParsers = {
  alert: parseAsString.withOptions({ history: "push" }),
  family: parseAsStringLiteral(ISSUE_FAMILIES),
  status: parseAsStringLiteral(ISSUE_STATUSES),
  type: parseAsStringLiteral(["usage", "model-deprecations"] as const),
  model: parseAsString,
  state: parseAsStringLiteral(ISSUE_STATUSES),
  usage_status: parseAsStringLiteral(ISSUE_STATUSES),
};
