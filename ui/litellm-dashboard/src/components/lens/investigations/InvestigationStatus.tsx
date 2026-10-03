"use client";

import { Button } from "@/components/ui/button";

import { ApiError } from "@/lib/http/client";

export function InvestigationError({
  error,
  queryError,
  refresh,
}: {
  error: string;
  queryError: unknown;
  refresh: () => void;
}) {
  const unavailable = queryError instanceof ApiError && queryError.status === 404;
  const queryMessage = queryError instanceof Error ? queryError.message : undefined;
  return (
    <div role="alert" className="rounded-lg border border-destructive/30 p-4 text-sm text-destructive">
      {unavailable
        ? "The Lens API is unavailable. Reload this page to use the current dashboard; if it persists, check the proxy deployment."
        : error || queryMessage}
      <Button variant="ghost" size="sm" onClick={() => (unavailable ? window.location.reload() : refresh())}>
        {unavailable ? "Reload page" : "Retry"}
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
