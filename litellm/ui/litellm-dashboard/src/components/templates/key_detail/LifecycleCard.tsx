"use client";

import type { ParseKeys } from "i18next";
import { Clock } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";

import DefaultProxyAdminTag from "@/components/common_components/DefaultProxyAdminTag";
import { userDetailHref } from "@/utils/entityLinks";
import { DEFAULT_PROXY_ADMIN_USER_ID } from "@/utils/sentinels";
import { cn } from "@/lib/cva.config";

import { formatDate, formatDateTime, formatRelative } from "./formatters";
import { keyLifecycle, type LifecycleEvent, type LifecycleSource } from "./keyLifecycle";
import { ResourceLink } from "./ResourceLink";
import { Caption, FactLabel, SectionCard } from "./SectionCard";

type DotState = "past" | "now" | "never" | "future" | "warn" | "moot" | "off";

export interface LifecycleCardProps {
  source: LifecycleSource;
  createdBy: { label: string; userId: string } | null;
  nowMs: number;
}

const DOT_CLASS: Readonly<Record<DotState, string>> = {
  past: "bg-muted-foreground",
  now: "bg-success ring-3 ring-success/25",
  never: "border-[1.5px] border-dashed border-muted-foreground bg-background",
  future: "border-[1.5px] border-muted-foreground bg-background",
  warn: "bg-warning",
  moot: "border-[1.5px] border-dashed border-muted-foreground bg-background",
  off: "border-[1.5px] border-muted-foreground bg-background",
};

const DASHED_STATES: ReadonlySet<DotState> = new Set(["now", "future", "never", "warn"]);

interface TimelineItem {
  readonly key: string;
  readonly state: DotState;
  readonly label: string;
  readonly when: string;
  readonly title?: string;
  readonly detail?: ReactNode;
  readonly warn?: boolean;
}

function CreatorLink({ createdBy }: { createdBy: { label: string; userId: string } }) {
  const { t } = useTranslation();
  if (createdBy.userId === DEFAULT_PROXY_ADMIN_USER_ID) return <DefaultProxyAdminTag userId={createdBy.userId} />;
  return (
    <>
      {t("keyDetail.overview.lifecycle.by")}
      <ResourceLink href={createdBy.userId ? userDetailHref(createdBy.userId) : undefined}>
        {createdBy.label}
      </ResourceLink>
    </>
  );
}

export function LifecycleCard({ source, createdBy, nowMs }: LifecycleCardProps) {
  const { t, i18n } = useTranslation();
  const locale = i18n.resolvedLanguage || "en";
  const label = (key: string) => t(`keyDetail.overview.lifecycle.${key}` as ParseKeys);
  const rotationDetail = (afterExpiry: boolean, interval: string | null): string | undefined => {
    if (afterExpiry) return label("afterExpiry");
    return interval ? t("keyDetail.overview.lifecycle.every", { interval }) : undefined;
  };
  const dated = (at: string) => ({ when: formatDate(at), title: formatDateTime(at) });

  const lastUsedItem = (at: string | null): TimelineItem =>
    at
      ? {
          key: "lastUsed",
          state: "now",
          label: label("lastUsed"),
          when: formatRelative(at, nowMs, locale),
          title: formatDateTime(at),
        }
      : { key: "lastUsed", state: "never", label: label("lastUsed"), when: label("never") };

  const expiresItem = (event: Extract<LifecycleEvent, { kind: "expires" }>): TimelineItem => {
    if (!event.at) return { key: event.kind, state: "future", label: label("expires"), when: label("never") };
    return {
      key: event.kind,
      state: event.soon || event.expired ? "warn" : "future",
      label: event.expired ? label("expired") : label("expires"),
      ...dated(event.at),
      detail: event.expired ? undefined : formatRelative(event.at, nowMs, locale),
      warn: event.soon,
    };
  };

  const toItem = (event: LifecycleEvent): TimelineItem => {
    switch (event.kind) {
      case "created":
        return {
          key: event.kind,
          state: "past",
          label: label("created"),
          ...(event.at ? dated(event.at) : { when: "-" }),
          detail:
            createdBy && (createdBy.label || createdBy.userId) ? <CreatorLink createdBy={createdBy} /> : undefined,
        };
      case "lastRotated":
        return { key: event.kind, state: "past", label: label("lastRotated"), ...dated(event.at) };
      case "lastUpdated":
        return { key: event.kind, state: "past", label: label("lastUpdated"), ...dated(event.at) };
      case "lastUsed":
        return lastUsedItem(event.at);
      case "expires":
        return expiresItem(event);
      case "nextRotation":
        return {
          key: event.kind,
          state: event.afterExpiry ? "moot" : "future",
          label: label("nextRotation"),
          ...(event.at ? dated(event.at) : { when: label("notScheduled") }),
          detail: rotationDetail(event.afterExpiry, event.interval),
        };
      case "autoRotationOff":
        return { key: event.kind, state: "off", label: label("autoRotation"), when: label("off") };
    }
  };

  const items = keyLifecycle(source, nowMs).map(toItem);

  return (
    <SectionCard icon={Clock} title={t("keyDetail.overview.lifecycle.title")} testId="key-lifecycle-card">
      <ol className="m-0 grid list-none grid-cols-2 gap-y-4 p-0 sm:grid-flow-col sm:auto-cols-fr sm:grid-cols-none sm:gap-y-0">
        {items.map((item) => (
          <TimelineStep key={item.key} item={item} />
        ))}
      </ol>
    </SectionCard>
  );
}

function TimelineStep({ item }: { item: TimelineItem }) {
  const dimmed = item.state === "off" || item.state === "moot";
  return (
    <li className="group relative min-w-0 pt-[18px] pr-4" data-state={item.state}>
      <span
        aria-hidden
        className={cn(
          "absolute top-1 right-0 left-[9px] hidden border-t border-border group-last:hidden sm:block",
          DASHED_STATES.has(item.state) && "border-dashed",
        )}
      />
      <span aria-hidden className={cn("absolute top-0 left-0 size-[9px] rounded-full", DOT_CLASS[item.state])} />
      <FactLabel>{item.label}</FactLabel>
      <div
        className={cn("mt-0.5 truncate font-medium tabular-nums", dimmed && "text-muted-foreground")}
        title={item.title}
      >
        {item.when}
      </div>
      {item.detail ? (
        <Caption className={cn("flex min-w-0 items-center gap-1", item.warn && "font-medium text-warning")}>
          {item.detail}
        </Caption>
      ) : null}
    </li>
  );
}
