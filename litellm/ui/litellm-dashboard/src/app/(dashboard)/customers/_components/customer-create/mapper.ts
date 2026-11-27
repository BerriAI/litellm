import {
  buildCustomerOrganizationBody,
  initialCustomerOrganizationFields,
  type CustomerOrganizationFields,
} from "../customerOrganizationForm";

export function customerCreateDefaults() {
  const { name, customerOrgId, groups, defaultGroupId, ...limits } = initialCustomerOrganizationFields();
  return { name, customerOrgId, limits, budgetGroups: { groups, defaultGroupId } };
}

export function toCustomerCreateBody({
  name,
  customerOrgId,
  limits,
  budgetGroups,
}: ReturnType<typeof customerCreateDefaults>) {
  const fields: CustomerOrganizationFields = { name, customerOrgId, ...limits, ...budgetGroups };
  return buildCustomerOrganizationBody(fields);
}
