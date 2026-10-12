import type { FilterFn } from "@tanstack/react-table";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

import { customerHasMcpAccess } from "./customerResources";

export const CUSTOMER_FILTER_FIELDS = [
  {
    id: "blocked",
    label: "Status",
    options: [
      { value: "active", label: "Active" },
      { value: "blocked", label: "Blocked" },
    ],
  },
  {
    id: "max_budget",
    label: "Budget",
    options: [
      { value: "yes", label: "Has budget" },
      { value: "no", label: "No budget" },
    ],
  },
  {
    id: "mcp_access",
    label: "MCP access",
    options: [
      { value: "yes", label: "Has MCP access" },
      { value: "no", label: "No MCP access" },
    ],
  },
] as const;

export const customerMatchesSearch = (customer: EndUser, search: string): boolean =>
  `${customer.user_id} ${customer.alias ?? ""}`.toLowerCase().includes(search.trim().toLowerCase());

export const customerMatchesFilter = (customer: EndUser, columnId: string, value: unknown): boolean => {
  if (columnId === "blocked") {
    if (value === "blocked") return Boolean(customer.blocked);
    if (value === "active") return !customer.blocked;
    return true;
  }
  if (value !== "yes" && value !== "no") return true;
  if (columnId === "max_budget") return (customer.litellm_budget_table?.max_budget != null) === (value === "yes");
  if (columnId === "mcp_access") return customerHasMcpAccess(customer) === (value === "yes");
  return true;
};

export const customerFilterFn: FilterFn<EndUser> = (row, columnId, value: unknown) =>
  customerMatchesFilter(row.original, columnId, value);

export const formatCustomerFilterValue = (columnId: string, value: unknown): string =>
  CUSTOMER_FILTER_FIELDS.find((field) => field.id === columnId)?.options.find((option) => option.value === value)
    ?.label ?? String(value);
