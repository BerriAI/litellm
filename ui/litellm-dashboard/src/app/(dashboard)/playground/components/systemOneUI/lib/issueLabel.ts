const TOP_LEVEL: Record<string, string> = {
  state: "Input",
  input: "Input",
  model: "Model",
  questions: "Questions",
  safety_identifier: "Safety identifier",
  root: "Request",
  syntax: "Request",
};

const QUESTION_FIELD: Record<string, string> = {
  name: "name",
  type: "answer type",
  instructions: "instructions",
  criteria: "options",
};

const humanize = (segment: PropertyKey): string => String(segment).replace(/_/g, " ");

export function issueLabel(path: readonly PropertyKey[]): string {
  const [head = "root", question, field] = path;
  if (head !== "questions" || question === undefined) {
    return TOP_LEVEL[String(head)] ?? humanize(head);
  }
  const title = typeof question === "number" ? `Question ${question + 1}` : `Question "${String(question)}"`;
  if (field === undefined) {
    return title;
  }
  return `${title} ${QUESTION_FIELD[String(field)] ?? humanize(field)}`;
}
