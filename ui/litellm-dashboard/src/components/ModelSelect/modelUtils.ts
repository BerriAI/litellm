import { ProxyModel } from "@/app/(dashboard)/hooks/models/useModels";

export interface GroupedModels {
  wildcard: ProxyModel[];
  regular: ProxyModel[];
}

export const splitWildcardModels = (models: ProxyModel[]): GroupedModels => {
  const wildcard: ProxyModel[] = [];
  const regular: ProxyModel[] = [];

  for (const model of models) {
    if (model.id.endsWith("/*")) {
      wildcard.push(model);
    } else {
      regular.push(model);
    }
  }

  return { wildcard, regular };
};
