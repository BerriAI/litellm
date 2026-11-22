"use client";

import { useTranslation } from "react-i18next";

import { TabsList, TabsTrigger } from "@/components/ui/tabs";

const TRIGGER_CLASS =
  "flex-none rounded-sm px-3 font-medium text-muted-foreground data-active:bg-muted data-active:text-foreground data-active:shadow-none dark:data-active:border-transparent dark:data-active:bg-muted";

export function KeyTabsList({ showAutoRouter }: { showAutoRouter: boolean }) {
  const { t } = useTranslation();
  return (
    <TabsList
      aria-label={t("keyDetail.keyInfoView.tabs.label")}
      className="h-8 max-w-full rounded-md border border-border bg-background p-0.5 group-data-horizontal/tabs:h-8"
    >
      <TabsTrigger value="overview" className={TRIGGER_CLASS}>
        {t("keyDetail.keyInfoView.tabs.overview")}
      </TabsTrigger>
      <TabsTrigger value="activity" className={TRIGGER_CLASS}>
        {t("shared.resourceActivity.title")}
      </TabsTrigger>
      <TabsTrigger value="savings" className={TRIGGER_CLASS}>
        {t("keyDetail.keyInfoView.tabs.savings")}
      </TabsTrigger>
      {showAutoRouter && (
        <TabsTrigger value="auto-router-usage" className={TRIGGER_CLASS}>
          <span className="sr-only sm:not-sr-only">{t("keyDetail.keyInfoView.tabs.autoRouterUsage")}</span>
          <span className="sm:hidden" aria-hidden>
            {t("keyDetail.keyInfoView.tabs.autoRouterShort")}
          </span>
        </TabsTrigger>
      )}
      <TabsTrigger value="settings" className={TRIGGER_CLASS}>
        {t("keyDetail.keyInfoView.tabs.settings")}
      </TabsTrigger>
    </TabsList>
  );
}
