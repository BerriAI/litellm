import Papa from "papaparse";

import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

const cell = (value: string | number | null | undefined): string | number => value ?? "";

export const buildCustomersCsvRows = (customers: EndUser[]): Record<string, string | number>[] =>
  customers.map((customer) => {
    const budget = customer.litellm_budget_table;
    const permission = customer.object_permission;
    return {
      "Customer Alias": cell(customer.alias),
      "Customer ID": customer.user_id,
      Status: customer.blocked ? "Blocked" : "Active",
      "Spend (USD)": customer.spend,
      "Budget ID": cell(customer.budget_id ?? budget?.budget_id),
      "Max Budget (USD)": cell(budget?.max_budget),
      "Soft Budget (USD)": cell(budget?.soft_budget),
      "Budget Duration": cell(budget?.budget_duration),
      "Budget Reset At": cell(budget?.budget_reset_at),
      "TPM Limit": cell(budget?.tpm_limit),
      "RPM Limit": cell(budget?.rpm_limit),
      "TPD Limit": cell(budget?.tpd_limit),
      "Max Parallel Requests": cell(budget?.max_parallel_requests),
      Models: (customer.models ?? []).join(", "),
      "MCP Servers": (permission?.mcp_servers ?? []).join(", "),
      "MCP Access Groups": (permission?.mcp_access_groups ?? []).join(", "),
      "MCP Toolsets": (permission?.mcp_toolsets ?? []).join(", "),
      "Allowed Model Region": cell(customer.allowed_model_region),
      "Default Model": cell(customer.default_model),
    };
  });

export const buildCustomersCsv = (customers: EndUser[]): string =>
  Papa.unparse(buildCustomersCsvRows(customers), { escapeFormulae: /^[=+\-@\t\r]/ });

export const exportCustomersToCsv = (customers: EndUser[]): number => {
  const blob = new Blob([buildCustomersCsv(customers)], { type: "text/csv;charset=utf-8;" });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `customers_export_${new Date().toISOString().split("T")[0]}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
  return customers.length;
};
