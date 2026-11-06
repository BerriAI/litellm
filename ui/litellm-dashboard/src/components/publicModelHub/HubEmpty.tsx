"use client";

import { LayoutGrid, SearchX, TriangleAlert } from "lucide-react";
import * as React from "react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import i18n from "@/i18n/i18n";

import { SECTION_TITLE } from "./hubFormat";

const noResultsTitle = (noun: string, query: string, filtered: boolean) => {
  if (!filtered) return i18n.t("publicModelHub.noResults.noneYet", { noun });
  return query
    ? i18n.t("publicModelHub.noResults.noMatchQuery", { noun, query })
    : i18n.t("publicModelHub.noResults.noMatchFilters", { noun });
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
  const { t } = useTranslation();
  return (
    <Empty>
      <EmptyHeader>
        <EmptyMedia variant="icon">{icon}</EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>{noResultsTitle(noun, query, Boolean(onClear))}</EmptyTitle>
        <EmptyDescription>
          {onClear ? t("publicModelHub.noResults.filteredHint") : t("publicModelHub.noResults.emptyHint")}
        </EmptyDescription>
      </EmptyHeader>
      {onClear && (
        <EmptyContent>
          <Button variant="outline" size="sm" onClick={onClear}>
            {t("publicModelHub.noResults.clearFilters")}
          </Button>
        </EmptyContent>
      )}
    </Empty>
  );
}

export function HubNotEnabled() {
  const { t } = useTranslation();
  return (
    <Empty className="rounded-xl border border-solid border-border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <LayoutGrid />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>{t("publicModelHub.notEnabled.title")}</EmptyTitle>
        <EmptyDescription>{t("publicModelHub.notEnabled.description")}</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}

export function LoadFailed({ noun, onRetry }: { noun: string; onRetry: () => void }) {
  const { t } = useTranslation();
  return (
    <Empty>
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <TriangleAlert />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>{t("publicModelHub.loadFailed.title", { noun })}</EmptyTitle>
        <EmptyDescription>{t("publicModelHub.loadFailed.description")}</EmptyDescription>
      </EmptyHeader>
      <EmptyContent>
        <Button variant="outline" size="sm" onClick={onRetry}>
          {t("publicModelHub.loadFailed.retry")}
        </Button>
      </EmptyContent>
    </Empty>
  );
}

export function NotListed({ name }: { name: string }) {
  const { t } = useTranslation();
  return (
    <Empty className="h-full">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <SearchX />
        </EmptyMedia>
        <EmptyTitle className={SECTION_TITLE}>{t("publicModelHub.notListed.title", { name })}</EmptyTitle>
        <EmptyDescription>{t("publicModelHub.notListed.description")}</EmptyDescription>
      </EmptyHeader>
    </Empty>
  );
}
