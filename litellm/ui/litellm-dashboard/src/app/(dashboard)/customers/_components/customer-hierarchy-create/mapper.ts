import {
  buildCustomerHierarchyCreate,
  newCustomerHierarchyFields,
  type CustomerHierarchyCreateTarget,
} from "../customerHierarchyForms";

export function hierarchyCreateDefaults(target: CustomerHierarchyCreateTarget) {
  const { alias, ...budget } = newCustomerHierarchyFields(target);
  return { alias, identifier: "", budget };
}

export function toHierarchyCreate(
  target: CustomerHierarchyCreateTarget,
  { alias, identifier, budget }: ReturnType<typeof hierarchyCreateDefaults>,
) {
  return buildCustomerHierarchyCreate(target, { alias, ...budget }, identifier);
}
