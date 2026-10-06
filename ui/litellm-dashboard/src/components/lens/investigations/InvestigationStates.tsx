"use client";

import { ArrowUpRight, SearchX, TriangleAlert } from "lucide-react";
import { LoadingState } from "@/components/shared/LoadingState";
import { StateMessage } from "@/components/shared/StateMessage";
import { Button, buttonVariants } from "@/components/ui/button";

import { ApiError } from "@/lib/http/client";

const DOCS_URL = "https://docs.litellm.ai/docs/proxy/lens";

function DocsLink() {
  return (
    <a
      href={DOCS_URL}
      target="_blank"
      rel="noopener noreferrer"
      className={buttonVariants({ variant: "ghost", size: "sm" })}
    >
      Lens docs
      <ArrowUpRight aria-hidden="true" className="size-3.5" />
    </a>
  );
}

function loadFailureMessage(queryError: unknown, unavailable: boolean): string {
  if (unavailable)
    return "This dashboard may be newer than the proxy. Reload the page, and if it persists, check the proxy deployment.";
  if (queryError instanceof Error) return queryError.message;
  return "Something went wrong while contacting the proxy.";
}

export function InvestigationsLoadFailed({ queryError, refresh }: { queryError: unknown; refresh: () => void }) {
  const unavailable = queryError instanceof ApiError && queryError.status === 404;
  return (
    <StateMessage
      role="alert"
      tone="destructive"
      icon={<TriangleAlert className="size-5" />}
      title={unavailable ? "Lens API is unavailable" : "Couldn't load investigations"}
      description={loadFailureMessage(queryError, unavailable)}
    >
      <Button size="sm" onClick={() => (unavailable ? window.location.reload() : refresh())}>
        {unavailable ? "Reload page" : "Try again"}
      </Button>
      <DocsLink />
    </StateMessage>
  );
}

export function InvestigationError({ message, refresh }: { message: string; refresh: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-center justify-between gap-3 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive"
    >
      <span className="min-w-0">{message}</span>
      <Button variant="ghost" size="sm" onClick={refresh}>
        Retry
      </Button>
    </div>
  );
}

export function InvestigationsLoading() {
  return (
    <LoadingState
      title="Loading investigations…"
      description="Fetching your investigations and their latest findings."
    />
  );
}

export function InvestigationMissing({ selectLens }: { selectLens: (id: string | null) => void }) {
  return (
    <StateMessage
      role="alert"
      icon={<SearchX className="size-5" />}
      title="Investigation not found"
      description="It may have been deleted, or the link points to a different proxy."
    >
      <Button size="sm" onClick={() => selectLens(null)}>
        View all investigations
      </Button>
    </StateMessage>
  );
}
