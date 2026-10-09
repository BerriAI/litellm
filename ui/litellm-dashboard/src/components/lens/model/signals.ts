import type { AnalysisModelInfo, SignalConfig } from "./types";

export const SYSTEM_ONE_MODE = "decisions";

export const isDecisionsModelMode = (mode: string | null | undefined): boolean =>
  mode === SYSTEM_ONE_MODE || mode === "evaluation";

export const signalsConfigured = (config: SignalConfig): boolean =>
  Boolean(config.model) && (config.signals?.length ?? 0) > 0;

export const systemOneModels = (details: readonly AnalysisModelInfo[]): AnalysisModelInfo[] =>
  details
    .filter((info) => isDecisionsModelMode(info.mode))
    .toSorted((a, b) => a.model_group.localeCompare(b.model_group));

export interface LibrarySignal {
  readonly id: string;
  readonly name: string;
  readonly summary: string;
  readonly question: string;
}

export const SIGNAL_LIBRARY: readonly LibrarySignal[] = [
  {
    id: "user_frustration",
    name: "User frustration",
    summary: "annoyed, complaining or giving up",
    question:
      "Does the user show frustration, annoyance or dissatisfaction with the agent in this run, for example complaints, irritated corrections, all caps, profanity, or giving up on the task?",
  },
  {
    id: "missing_capability",
    name: "Missing capability",
    summary: "asked for something the agent can't do",
    question:
      "Does the user ask for something the agent cannot do in this run, so that the agent refuses, says it lacks a tool, permission, integration or data source, or fails because the capability does not exist?",
  },
  {
    id: "repeated_request",
    name: "Repeated request",
    summary: "had to ask for the same thing again",
    question:
      "Does the user ask for the same thing more than once in this run, usually because the agent did not deliver it the first time?",
  },
  {
    id: "asked_for_human",
    name: "Asked for a human",
    summary: "wants a person, not the agent",
    question: "Does the user ask to talk to a human, a support person or a manager instead of the agent in this run?",
  },
  {
    id: "tool_failure",
    name: "Tool failure",
    summary: "a tool or step failed and stayed broken",
    question:
      "Does a tool call or step fail in this run with an error, exception or timeout that the agent does not recover from?",
  },
  {
    id: "refused_request",
    name: "Refused request",
    summary: "declined a reasonable ask",
    question: "Does the agent refuse or decline a reasonable, allowed user request in this run?",
  },
  {
    id: "made_up_answer",
    name: "Made-up answer",
    summary: "facts or links with no source",
    question:
      "Does the agent state facts, numbers, links or tool results in this run that are not supported by the conversation or by any tool output?",
  },
  {
    id: "task_abandoned",
    name: "Task abandoned",
    summary: "run ended without what was asked",
    question: "Does the run end without the user getting what they asked for?",
  },
];
