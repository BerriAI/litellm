"use client";

import { LayoutGrid, SearchX, TriangleAlert } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";

import { SECTION_TITLE } from "./hubFormat";

const noResultsTitle = (noun: string, query: string, filtered: boolean) => {
  if (!filtered) return `No ${noun} published yet`;
  return query ? `No ${noun} match "${query}"` : `No ${noun} match these filters`;
};

export function NoResults({
  icon,
  noun,
  query,
  onClear,
}: {
  icon: React.ReactNode;
  noun: string;
  query: string;
  onClear?: () => void;
}) {
  return (
    <Empty>
      <EmptyHeader>
        <EmptyMedia variant="icon">{icon}</EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>{noResultsTitle(noun, query, Boolean(onClear))}</EmptyTitle>
        <EmptyDescription>
          {onClear ? "Try a shorter search or fewer filters" : "Your admin publishes them from the AI Hub page"}
        </EmptyDescription>
      </EmptyHeader>
      {onClear && (
        <EmptyContent>
          <Button variant="outline" size="sm" onClick={onClear}>
            Clear filters
          </Button>
        </EmptyContent>
      )}
    </Empty>
  );
}

export function HubNotEnabled() {
  return (
    <Empty className="rounded-xl border border-solid border-border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <LayoutGrid />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>This AI Hub isn&apos;t available</EmptyTitle>
        <EmptyDescription>Ask your proxy admin to make it public from their Admin UI</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

export function LoadFailed({ noun, onRetry }: { noun: string; onRetry: () => void }) {
  return (
    <Empty>
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <TriangleAlert />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>Couldn&apos;t load {noun}</EmptyTitle>
        <EmptyDescription>The gateway returned an error. Try again in a moment</EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        <Button variant="outline" size="sm" onClick={onRetry}>
          Try again
        </Button>
      </EmptyContent>
    </Empty>
  );
}

export function NotListed({ name }: { name: string }) {
  return (
    <Empty className="h-full">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <SearchX />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>&ldquo;{name}&rdquo; isn&apos;t listed here</EmptyTitle>
        <EmptyDescription>It may have been renamed or unpublished</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}
