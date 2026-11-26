import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { buildUpdateCustomerBody, type EditCustomerFields, type UpdateCustomerBody } from "./customerForms";
import type { EditCustomerFormValues } from "./editCustomerSchema";

const asText = (n: number | null | undefined) => (n === null || n === undefined ? "" : String(n));

export const customerToForm = (customer: EndUser): EditCustomerFormValues => ({
  alias: customer.alias ?? "",
  budget: asText(customer.litellm_budget_table?.max_budget),
});

export const toUpdateCustomerBody = (customer: EndUser, values: EditCustomerFormValues): UpdateCustomerBody => {
  const fields: EditCustomerFields = {
    userId: customer.user_id,
    alias: values.alias,
    budget: values.budget,
    blocked: Boolean(customer.blocked),
  };
  return buildUpdateCustomerBody(fields);
};
