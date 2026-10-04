"use client";

import type { ReactNode } from "react";
import { ArrowUpRight, Loader2, SearchX, TriangleAlert } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

import { ApiError } from "@/lib/http/client";

const DOCS_URL = "https://docs.litellm.ai/docs/proxy/lens";

function CenteredState({
  role,
  icon,
  tone = "muted",
  title,
  description,
  children,
}: {
  role: "status" | "alert";
  icon: ReactNode;
  tone?: "muted" | "destructive";
  title: string;
  description: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div
      role={role}
      className="m-auto flex max-w-sm flex-col items-center gap-3 py-16 text-center animate-in fade-in-0 duration-300 motion-reduce:animate-none"
    >
      <span
        aria-hidden="true"
        className={cn(
          "flex size-10 items-center justify-center rounded-full",
          tone === "destructive" ? "bg-destructive/10 text-destructive" : "bg-muted text-muted-foreground",
        )}
      >
        {icon}
      </span>
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium text-foreground">{title}</p>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      {children && <div className="mt-1 flex items-center gap-2">{children}</div>}
    </div>
  );
}

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
    <CenteredState
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
    </CenteredState>
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
    <CenteredState
      role="status"
      icon={<Loader2 className="size-5 animate-spin motion-reduce:animate-none" />}
      title="Loading investigations…"
      description="Fetching your investigations and their latest findings."
    />
  );
}

export function InvestigationMissing({ selectLens }: { selectLens: (id: string | null) => void }) {
  return (
    <CenteredState
      role="alert"
      icon={<SearchX className="size-5" />}
      title="Investigation not found"
      description="It may have been deleted, or the link points to a different proxy."
    >
      <Button size="sm" onClick={() => selectLens(null)}>
        View all investigations
      </Button>
    </CenteredState>
  );
}
