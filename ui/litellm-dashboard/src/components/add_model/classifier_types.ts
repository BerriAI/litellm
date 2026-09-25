export type ClassifierType = "heuristic" | "heuristic_v2" | "llm" | "jev" | "heuristic_first" | "hybrid";

export const usesLlmClassifier = (classifierType: ClassifierType): boolean =>
  (["llm", "heuristic_first", "hybrid"] as const).some((type) => type === classifierType);

export const usesClassifierContext = (classifierType: ClassifierType): boolean =>
  classifierType === "jev" || usesLlmClassifier(classifierType);
