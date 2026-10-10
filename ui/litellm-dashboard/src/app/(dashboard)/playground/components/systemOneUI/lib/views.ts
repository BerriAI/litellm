import {
  isOpenAIDecisionsRequest,
  isOpenAIDecisionsResponse,
  type OpenAIDecisionAnswer,
  type OpenAIDecisionQuestion,
  type PlaygroundQuestion,
  type PlaygroundRequest,
  type PlaygroundResponse,
  type SystemOneAnswer,
} from "./schemas";

export interface ProbabilityView {
  readonly label: string;
  readonly probability: number;
  readonly selected: boolean;
}

export type AnswerView =
  | { readonly id: string; readonly type: "noul"; readonly probability: number }
  | { readonly id: string; readonly type: "predicate"; readonly probability: number }
  | {
      readonly id: string;
      readonly type: "choice";
      readonly choice: string;
      readonly confidence?: number;
      readonly probabilities: readonly ProbabilityView[];
    }
  | {
      readonly id: string;
      readonly type: "score";
      readonly score: number;
      readonly confidence?: number;
      readonly probabilities: readonly ProbabilityView[];
    }
  | { readonly id: string; readonly type: "refusal" };

export interface CriterionView {
  readonly label: string;
  readonly description?: string;
}

export interface QuestionView {
  readonly id: string;
  readonly type: string;
  readonly instructions?: string;
  readonly ordered: boolean;
  readonly criteria?: readonly CriterionView[];
}

export interface RequestView {
  readonly contextLabel: "State" | "Input";
  readonly context: string;
  readonly questions: readonly QuestionView[];
}

export function formatJson(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  return JSON.stringify(value, null, 2) ?? String(value);
}

const positionalId = (name: string | null | undefined, index: number): string => name ?? `q${index}`;

const legendLabel = (level: string, description: unknown): string =>
  description === undefined
    ? level
    : `${level}: ${typeof description === "string" ? description : JSON.stringify(description)}`;

function systemOneAnswerView(id: string, answer: SystemOneAnswer): AnswerView {
  if (answer.type === "noul") {
    return { id, type: "noul", probability: answer.noul };
  }
  if (answer.type === "choice") {
    const probabilities = Object.entries(answer.probabilities)
      .sort(([, first], [, second]) => second - first)
      .map(([label, probability]) => ({ label, probability, selected: label === answer.choice }));
    return { id, type: "choice", choice: answer.choice, confidence: answer.confidence, probabilities };
  }
  const probabilities = Object.entries(answer.probabilities)
    .sort(([first], [second]) => Number(first) - Number(second))
    .map(([level, probability]) => ({
      label: legendLabel(level, answer.legend?.[level]),
      probability,
      selected: Number(level) === Math.round(answer.score),
    }));
  return { id, type: "score", score: answer.score, confidence: answer.confidence, probabilities };
}

function openAIAnswerView(answer: OpenAIDecisionAnswer, index: number): AnswerView {
  const id = positionalId(answer.name, index);
  if (answer.type === "predicate") {
    return { id, type: "predicate", probability: answer.probability };
  }
  if (answer.type === "refusal") {
    return { id, type: "refusal" };
  }
  if (answer.type === "choice") {
    const choice = String(answer.choice);
    const probabilities = [...answer.probabilities]
      .sort((first, second) => second.probability - first.probability)
      .map((option) => ({
        label: String(option.value),
        probability: option.probability,
        selected: option.value === answer.choice,
      }));
    return { id, type: "choice", choice, confidence: answer.confidence, probabilities };
  }
  const probabilities = [...answer.probabilities]
    .sort((first, second) => first.value - second.value)
    .map((level) => ({
      label: `${level.value}: ${level.label}`,
      probability: level.probability,
      selected: level.value === Math.round(answer.score),
    }));
  return { id, type: "score", score: answer.score, confidence: answer.confidence, probabilities };
}

export const answerViews = (response: PlaygroundResponse): readonly AnswerView[] =>
  isOpenAIDecisionsResponse(response)
    ? response.answers.map(openAIAnswerView)
    : Object.entries(response.answers).map(([id, answer]) => systemOneAnswerView(id, answer));

const definedDescription = (description: unknown): string | undefined =>
  description === undefined || description === null ? undefined : formatJson(description);

function systemOneQuestionView(id: string, question: PlaygroundQuestion): QuestionView {
  const instructions = definedDescription(question.instructions);
  if (question.type === "choice") {
    const criteria = Object.entries(question.criteria).map(([label, description]) => ({
      label,
      description: definedDescription(description) ?? "",
    }));
    return { id, type: question.type, instructions, ordered: false, criteria };
  }
  if (question.type === "noul") {
    const criteria = Object.entries(question.criteria ?? {})
      .filter(([, description]) => description !== undefined)
      .map(([label, description]) => ({ label, description: definedDescription(description) ?? "" }));
    return { id, type: question.type, instructions, ordered: false, criteria };
  }
  const criteria = question.criteria.map((description, index) => ({
    label: String(index),
    description: formatJson(description),
  }));
  return { id, type: question.type, instructions, ordered: true, criteria };
}

function openAIQuestionView(question: OpenAIDecisionQuestion, index: number): QuestionView {
  const id = positionalId(question.name, index);
  if (question.type === "predicate") {
    return { id, type: question.type, instructions: question.instructions, ordered: false };
  }
  if (question.type === "choice") {
    const criteria = question.choices.map((option) => ({
      label: String(option.value),
      description: option.description ?? undefined,
    }));
    return { id, type: question.type, instructions: question.instructions, ordered: false, criteria };
  }
  const criteria = question.levels.map((level, position) => ({
    label: String(position),
    description: level.description ? `${level.label}: ${level.description}` : level.label,
  }));
  return { id, type: question.type, instructions: question.instructions, ordered: true, criteria };
}

export const requestView = (payload: PlaygroundRequest): RequestView =>
  isOpenAIDecisionsRequest(payload)
    ? {
        contextLabel: "Input",
        context: formatJson(payload.input),
        questions: payload.questions.map(openAIQuestionView),
      }
    : {
        contextLabel: "State",
        context: formatJson(payload.state),
        questions: Object.entries(payload.questions).map(([id, question]) => systemOneQuestionView(id, question)),
      };
