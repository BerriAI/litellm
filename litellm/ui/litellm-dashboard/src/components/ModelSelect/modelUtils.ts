export interface GroupedModels {
  wildcard: string[];
  regular: string[];
}

export const splitWildcardModels = (models: string[]): GroupedModels => {
  const wildcard: string[] = [];
  const regular: string[] = [];

  for (const model of models) {
    if (model.endsWith("/*")) {
      wildcard.push(model);
    } else {
      regular.push(model);
    }
  }

  return { wildcard, regular };
};
