"use client";

import { useQuery } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { fetchClient } from "@/lib/http/api";
import { all_admin_roles } from "@/utils/roles";

import { DEFAULT_POOL_NAME } from "./workloadClass";

export const WORKLOAD_CLASS_QUERY_KEY = ["fairnessWorkloadClasses"] as const;

export const WORKLOAD_CLASS_HINT =
  "Fairness under load class. Reserved capacity and queue deadlines come from the class. A team's class wins over its keys.";

const fetchWorkloadClassNames = async (): Promise<readonly string[]> => {
  const { data } = await fetchClient.GET("/fairness/settings");
  return data?.settings.workload_classes?.map((cls) => cls.name) ?? [];
};

export const useWorkloadClassNames = (fetchNames: () => Promise<readonly string[]> = fetchWorkloadClassNames) => {
  const { userRole } = useAuthorized();
  const queryOptions = {
    queryKey: WORKLOAD_CLASS_QUERY_KEY,
    queryFn: fetchNames,
    enabled: all_admin_roles.includes(userRole),
    staleTime: 60_000,
    retry: false,
  };
  return useQuery(queryOptions);
};

export const useShowWorkloadClass = (current: string | undefined): boolean => {
  const { data: names = [] } = useWorkloadClassNames();
  return names.length > 0 || (current !== undefined && current !== DEFAULT_POOL_NAME);
};

export interface WorkloadClassSelectProps {
  readonly id?: string;
  readonly value: string | undefined;
  readonly onChange: (value: string) => void;
  readonly disabled?: boolean;
  readonly fetchNames?: () => Promise<readonly string[]>;
}

export function WorkloadClassSelect({ id, value, onChange, disabled, fetchNames }: WorkloadClassSelectProps) {
  const { data: names = [] } = useWorkloadClassNames(fetchNames);
  const current = value && value.length > 0 ? value : DEFAULT_POOL_NAME;
  const options = Array.from(new Set([DEFAULT_POOL_NAME, ...names, current]));
  return (
    <Select
      items={options.map((name) => ({ value: name, label: name }))}
      value={current}
      onValueChange={(next: string | null) => next != null && onChange(next)}
      disabled={disabled}
    >
      <SelectTrigger id={id} className="w-full" aria-label="Workload class">
        <SelectValue placeholder="Select workload class" />
      </SelectTrigger>
      <SelectContent>
        {options.map((name) => (
          <SelectItem key={name} value={name}>
            {name === DEFAULT_POOL_NAME ? "default (shared pool)" : name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
