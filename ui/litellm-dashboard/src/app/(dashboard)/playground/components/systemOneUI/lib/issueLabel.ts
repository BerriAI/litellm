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

const humanize = (segment: string): string => segment.replace(/_/g, " ");

export function issueLabel(path: string): string {
  const [head, question, field] = path.split(".");
  if (head !== "questions" || question === undefined) {
    return TOP_LEVEL[head] ?? humanize(head);
  }
  const title = /^\d+$/.test(question) ? `Question ${Number(question) + 1}` : `Question "${question}"`;
  if (field === undefined) {
    return title;
  }
  return `${title} ${QUESTION_FIELD[field] ?? humanize(field)}`;
}
