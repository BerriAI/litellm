"use client";

import { Button } from "@/components/ui/button";

import { ApiError } from "@/lib/http/client";

export function InvestigationError({
  error,
  query,
  refresh,
}: {
  error: string;
  query: import("@tanstack/react-query").UseQueryResult<import("../model/types").LensList, Error>;
  refresh: () => void;
}) {
  return (
    <div role="alert" className="rounded-lg border border-destructive/30 p-4 text-sm text-destructive">
      {query.error instanceof ApiError && query.error.status === 404
        ? "The Lens API is unavailable. Reload this page to use the current dashboard; if it persists, check the proxy deployment."
        : error || query.error?.message}
      <Button
        variant="ghost"
        size="sm"
        onClick={() =>
          query.error instanceof ApiError && query.error.status === 404 ? window.location.reload() : refresh()
        }
      >
        {query.error instanceof ApiError && query.error.status === 404 ? "Reload page" : "Retry"}
      </Button>
    </div>
  );
}
export function InvestigationsLoading() {
  return (
    <p role="status" className="py-8 text-sm text-muted-foreground">
      Loading investigations…
    </p>
  );
}
export function InvestigationMissing({ selectLens }: { selectLens: (id: string | null) => void }) {
  return (
    <div role="alert" className="text-sm">
      This investigation was not found.{" "}
      <Button variant="link" onClick={() => selectLens(null)}>
        View all investigations
      </Button>
    </div>
  );
}
