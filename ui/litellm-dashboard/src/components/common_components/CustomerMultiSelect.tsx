import { useMemo, useState } from "react";
import { useCustomerLookups, useCustomers } from "@/app/(dashboard)/hooks/customers/useCustomers";
import { PaginatedMultiSelect } from "@/components/shared/PaginatedMultiSelect";
import type { SearchSelectOption } from "@/components/shared/SearchSelect";

interface CustomerMultiSelectProps {
  value: readonly string[];
  onChange: (value: string[]) => void;
  disabled?: boolean;
}

export default function CustomerMultiSelect({ value, onChange, disabled }: CustomerMultiSelectProps) {
  const [search, setSearch] = useState("");
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, isLoading, isError } = useCustomers(20, search);
  const loadedCustomers = useMemo(() => (data?.pages ?? []).flat(), [data]);
  const missingIds = value.filter((id) => !loadedCustomers.some((customer) => customer.user_id === id));
  const selectedCustomers = useCustomerLookups(missingIds);
  const options: SearchSelectOption[] = Array.from(
    new Map(
      [...loadedCustomers, ...selectedCustomers.flatMap((result) => (result.data ? [result.data] : []))].map(
        (customer) =>
          [
            customer.user_id,
            { value: customer.user_id, label: customer.alias || customer.user_id, sublabel: customer.user_id },
          ] as const,
      ),
    ).values(),
  );
  const lookupError = selectedCustomers.some((result) => result.isError)
    ? "Unable to load selected customers"
    : undefined;
  const errorText = isError ? "Unable to load customers" : lookupError;

  return (
    <div>
      <PaginatedMultiSelect
        options={options}
        value={value}
        onValueChange={onChange}
        onSearchChange={setSearch}
        onLoadMore={fetchNextPage}
        hasNextPage={hasNextPage}
        isLoading={isLoading}
        isFetchingNextPage={isFetchingNextPage}
        placeholder="Search customers by ID or alias..."
        emptyText="No customers found"
        errorText={errorText}
        loadingText="Loading customers..."
        clearAllLabel="Clear all customers"
        disabled={disabled}
      />
      {errorText && (
        <p role="alert" className="mt-1 text-sm text-destructive">
          {errorText}
        </p>
      )}
    </div>
  );
}
