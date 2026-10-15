export type ClassifierType =
  | "heuristic"
  | "heuristic_v2"
  | "llm"
  | "jev"
  | "heuristic_first"
  | "hybrid"
  | "capability"
  | "llm_v2"
  | "custom";

export type LocalHeuristic = "heuristic" | "heuristic_v2";

export const isHeuristicChain = (classifierType: ClassifierType): boolean =>
  classifierType === "heuristic_first" || classifierType === "hybrid";

export const usesLlmClassifier = (classifierType: ClassifierType): boolean =>
  (["llm", "heuristic_first", "hybrid", "capability", "llm_v2"] as const).some((type) => type === classifierType);

export const usesClassifierContext = (classifierType: ClassifierType): boolean =>
  classifierType === "jev" || usesLlmClassifier(classifierType);
