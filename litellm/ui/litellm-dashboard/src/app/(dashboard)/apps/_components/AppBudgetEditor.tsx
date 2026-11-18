"use client";

import CustomerHierarchyBudgetFields from "../../customers/_components/CustomerHierarchyBudgetFields";
import type { CustomerHierarchyFields } from "../../customers/_components/customerHierarchyForms";

export default function AppBudgetEditor({
  title,
  fields,
  onChange,
  id,
}: {
  title: string;
  fields: CustomerHierarchyFields;
  onChange: (fields: CustomerHierarchyFields) => void;
  id: string;
}) {
  return (
    <fieldset className="min-w-0 space-y-3 rounded-lg border p-4">
      <legend className="px-1 text-sm font-semibold">{title}</legend>
      <CustomerHierarchyBudgetFields
        kind="organization"
        operation="edit"
        idPrefix={id}
        fields={fields}
        onChange={(key, value) => onChange({ ...fields, [key]: value })}
      />
    </fieldset>
  );
}
