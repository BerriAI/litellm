"use client";

import { Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button } from "@/components/ui/button";
import { Page } from "@/components/shared/Page";
import { PageHeader, PageHeaderControls, PageHeaderTitle } from "@/components/shared/PageHeader";
import { uiHref } from "@/utils/uiHref";
import { Leaderboard, useBreakdown } from "@/app/(dashboard)/usage/_components/components/overview/BreakdownChart";
import { Panel } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";
import UsageStatStrip from "@/app/(dashboard)/usage/_components/components/overview/UsageStatStrip";
import { HOME_USAGE_DAYS, useHomeUsage } from "./useHomeUsage";

const LEADERBOARD = { metric: "tokens", dimension: "model_groups" } as const;

export default function HomePage() {
  const { t } = useTranslation();
  const { isViewOnly } = useAuthorized();
  const usage = useHomeUsage();
  const { series, ranking } = useBreakdown(usage.results, LEADERBOARD, 8);

  return (
    <Page>
      <PageHeader>
        <PageHeaderTitle>{t("home.page.title")}</PageHeaderTitle>
        {!isViewOnly && (
          <PageHeaderControls>
            <Button render={<a href={uiHref("api-keys?create=true")} />}>
              <Plus />
              {t("home.page.createKey")}
            </Button>
          </PageHeaderControls>
        )}
      </PageHeader>

      <section aria-label={t("home.page.usageAria", { days: HOME_USAGE_DAYS })}>
        {usage.failed ? (
          <div className="flex items-center justify-between gap-2 rounded-xl border bg-card px-4 py-3 text-sm">
            <span className="text-destructive">{t("home.page.usageFailed")}</span>
            <Button variant="outline" size="sm" onClick={usage.retry}>
              {t("home.page.retry")}
            </Button>
          </div>
        ) : (
          <UsageStatStrip
            results={usage.results}
            totals={usage.totals}
            loading={usage.loading}
            requestCountsPending={usage.requestCountsPending}
            budget={usage.budget}
          />
        )}
      </section>

      <Panel
        title={t("home.page.leaderboard.title")}
        subtitle={t("home.page.leaderboard.subtitle", { days: HOME_USAGE_DAYS })}
        testId="home-leaderboard"
      >
        <Leaderboard
          ranking={ranking}
          series={series}
          metric={LEADERBOARD.metric}
          dimension={LEADERBOARD.dimension}
          columns={2}
          limit={10}
        />
      </Panel>
    </Page>
  );
}
