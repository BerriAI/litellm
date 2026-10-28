import type { DataTableSortField } from "@/components/shared/DataTable";

export const KEY_RATE_LIMIT_SORT_FIELDS: DataTableSortField[] = [
  { id: "tpm_limit", label: "TPM" },
  { id: "rpm_limit", label: "RPM" },
];
