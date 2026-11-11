import type { ParseKeys, TFunction } from "i18next";
import i18n from "@/i18n/i18n";

const TOP_LEVEL: Record<string, ParseKeys> = {
  state: "playground.systemOne.issueLabel.input",
  input: "playground.systemOne.issueLabel.input",
  model: "playground.systemOne.issueLabel.model",
  questions: "playground.systemOne.issueLabel.questions",
  safety_identifier: "playground.systemOne.issueLabel.safetyIdentifier",
  root: "playground.systemOne.issueLabel.request",
  syntax: "playground.systemOne.issueLabel.request",
};

const QUESTION_FIELD: Record<string, ParseKeys> = {
  name: "playground.systemOne.issueLabel.fields.name",
  type: "playground.systemOne.issueLabel.fields.type",
  instructions: "playground.systemOne.issueLabel.fields.instructions",
  criteria: "playground.systemOne.issueLabel.fields.criteria",
};

const humanize = (segment: PropertyKey): string => String(segment).replace(/_/g, " ");

export function issueLabel(path: readonly PropertyKey[], t: TFunction = i18n.t.bind(i18n)): string {
  const [head = "root", question, field] = path;
  if (head !== "questions" || question === undefined) {
    return Object.hasOwn(TOP_LEVEL, String(head)) ? t(TOP_LEVEL[String(head)]) : humanize(head);
  }
  const title =
    typeof question === "number"
      ? t("playground.systemOne.issueLabel.question", { number: question + 1 })
      : t("playground.systemOne.issueLabel.namedQuestion", { name: String(question) });
  if (field === undefined) {
    return title;
  }
  return `${title} ${Object.hasOwn(QUESTION_FIELD, String(field)) ? t(QUESTION_FIELD[String(field)]) : humanize(field)}`;
}
