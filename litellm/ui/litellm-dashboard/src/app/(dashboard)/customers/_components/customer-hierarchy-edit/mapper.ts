import {
  buildCustomerHierarchyPatch,
  customerHierarchyFields,
  type CustomerHierarchyEntity,
  type Organization,
} from "../customerHierarchyForms";

export function hierarchyEditDefaults(entity: CustomerHierarchyEntity, organization: Organization) {
  const { alias, ...budget } = customerHierarchyFields(entity, organization);
  return { alias, budget };
}

export function toHierarchyPatch(
  entity: CustomerHierarchyEntity,
  organization: Organization,
  { alias, budget }: ReturnType<typeof hierarchyEditDefaults>,
) {
  return buildCustomerHierarchyPatch(entity, { alias, ...budget }, organization);
}
