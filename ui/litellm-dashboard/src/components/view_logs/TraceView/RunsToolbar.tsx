"use client";

import { Search } from "lucide-react";

import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

export type RunStatusFilter = "all" | "ok" | "error";

export const ALL_SERVICES = "all";

interface RunsToolbarProps {
  query: string;
  service: string;
  status: RunStatusFilter;
  services: string[];
  onQueryChange: (value: string) => void;
  onServiceChange: (value: string) => void;
  onStatusChange: (value: RunStatusFilter) => void;
  /** Extra controls (time range, live tail) rendered on the right. */
  children?: React.ReactNode;
}

const STATUS_ITEMS: { value: RunStatusFilter; label: string }[] = [
  { value: "all", label: "All status" },
  { value: "ok", label: "Succeeded" },
  { value: "error", label: "Failed" },
];

/** Search + service / status filters for the Runs table. Filtering is client-side over the loaded page. */
export function RunsToolbar({
  query,
  service,
  status,
  services,
  onQueryChange,
  onServiceChange,
  onStatusChange,
  children,
}: RunsToolbarProps) {
  const serviceItems = [
    { value: ALL_SERVICES, label: "All services" },
    ...services.map((s) => ({ value: s, label: s })),
  ];
  return (
    <div className="flex min-h-11 shrink-0 flex-wrap items-center gap-2 border-b border-border bg-card px-3 py-2">
      <div className="relative w-full max-w-[380px]">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Search input or trace ID"
          aria-label="Search runs"
          className="h-7 pl-8 text-[12px]"
        />
      </div>
      <Select
        items={serviceItems}
        value={service}
        onValueChange={(value: string | null) => value !== null && onServiceChange(value)}
      >
        <SelectTrigger size="sm" className="h-7 min-w-[130px] text-[12px]" aria-label="Filter by service">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {serviceItems.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select
        items={STATUS_ITEMS}
        value={status}
        onValueChange={(value: RunStatusFilter | null) => value !== null && onStatusChange(value)}
      >
        <SelectTrigger size="sm" className="h-7 min-w-[110px] text-[12px]" aria-label="Filter by status">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {STATUS_ITEMS.map((item) => (
            <SelectItem key={item.value} value={item.value}>
              {item.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {children && <div className="ml-auto flex items-center gap-2">{children}</div>}
    </div>
  );
}
