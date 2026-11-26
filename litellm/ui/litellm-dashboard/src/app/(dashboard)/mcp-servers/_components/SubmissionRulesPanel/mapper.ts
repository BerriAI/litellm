import { type SubmissionRulesValues } from "./schema";

export const toSubmissionRulesValues = (requiredFields: string[]): SubmissionRulesValues => ({ requiredFields });

export const toggleRequiredField = (requiredFields: string[], key: string): string[] =>
  requiredFields.includes(key) ? requiredFields.filter((field) => field !== key) : [...requiredFields, key];
