export type ClassifierType = "heuristic" | "llm" | "jev";

export const usesLlmClassifier = (classifierType: ClassifierType): boolean => classifierType === "llm";

export const usesClassifierContext = (classifierType: ClassifierType): boolean =>
  classifierType === "jev" || usesLlmClassifier(classifierType);
