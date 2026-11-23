"use client";

import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import type { ParseKeys } from "i18next";
import { ArrowRight, Bell, CornerDownRight, Info, Wallet } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { budgetUsagePercent, type AppliedBudget } from "@/components/shared/appliedBudgets";
import { StatusBadge } from "@/components/shared/table_cells";
import type { BudgetAlertRow } from "@/components/key_team_helpers/budgetAlertEmails";
import { budgetPeriodRange, formatBudgetResetTime } from "@/utils/budgetUtils";
import { cn } from "@/lib/cva.config";

import { headroomKey, meterTone, ownerText, resetCaption, resetSentence, SCOPE_KEYS } from "./budgetText";
import { budgetThrottles, canStillSpend } from "./canStillSpend";
import { formatDate, formatMoney } from "./formatters";
import { ResourceLink } from "./ResourceLink";
import { Caption, SectionCard, SubHead } from "./SectionCard";

dayjs.extend(utc);

const TWO_DAYS_MS = 2 * 24 * 60 * 60 * 1000;

export interface SpendCardProps {
  spend: number;
  totalSpend: number | null | undefined;
  lifetimeNeedsBackfill: boolean;
  budgetDuration: string | null;
  budgetResetAt: string | null;
  createdAt: string | null;
  hasOwnBudget: boolean;
  budgets: readonly AppliedBudget[];
  keyThrottlesOnBudget: boolean;
  alerts: readonly BudgetAlertRow[];
  fallbacks: Readonly<Record<string, readonly string[]>>;
  budgetHref: (budget: AppliedBudget) => string | undefined;
  onOpenActivity: () => void;
}

export function SpendCard(props: SpendCardProps) {
  const { t } = useTranslation();
  const footer = [
    props.alerts.length > 0 ? <AlertsLine key="alerts" alerts={props.alerts} /> : null,
    Object.keys(props.fallbacks).length > 0 ? <FallbacksLine key="fallbacks" fallbacks={props.fallbacks} /> : null,
  ].filter((line) => line !== null);

  return (
    <SectionCard
      icon={Wallet}
      title={t("keyDetail.overview.spend.title")}
      testId="key-spend-card"
      action={
        <Button variant="outline" size="sm" onClick={props.onOpenActivity}>
          {t("keyDetail.overview.spend.openActivity")}
        </Button>
      }
      footer={footer.length > 0 ? footer : undefined}
    >
      <div className="grid grid-cols-1 items-start gap-6 md:grid-cols-[minmax(180px,240px)_minmax(0,1fr)]">
        <div className="flex flex-col gap-3">
          <SpendHeadline {...props} />
          <LifetimeStat {...props} />
        </div>
        <CanStillSpendSlot {...props} />
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <SubHead title={t("shared.appliedBudgets.title")} count={props.budgets.length} />
        {props.budgets.length > 0 ? (
          <ul className="m-0 list-none p-0" data-testid="applied-budgets">
            {props.budgets.map((budget) => (
              <BudgetRow key={budget.scope} budget={budget} {...props} />
            ))}
          </ul>
        ) : (
          <p className="m-0 border-t border-border py-2.5 text-muted-foreground">{t("shared.appliedBudgets.empty")}</p>
        )}
      </div>
    </SectionCard>
  );
}

function periodHint(budgetDuration: string | null, budgetResetAt: string | null): string | null {
  const period = budgetPeriodRange(budgetDuration, budgetResetAt);
  if (!period) return null;
  const start = dayjs.utc(period.start);
  const end = dayjs.utc(period.end);
  if (end.diff(start) < TWO_DAYS_MS) return null;
  return `${start.format("MMM D")}|${end.subtract(1, "millisecond").format("MMM D")}|${end.format("MMM D, YYYY")}`;
}

function SpendHeadline({ spend, budgetDuration, budgetResetAt, hasOwnBudget }: SpendCardProps) {
  const { t } = useTranslation();
  const resetTime = formatBudgetResetTime(budgetResetAt);
  const range = periodHint(budgetDuration, budgetResetAt);
  const [start, end, reset] = range ? range.split("|") : [];
  const hint = (() => {
    if (range) return t("keyDetail.overview.spend.periodRange", { start, end, reset });
    if (resetTime) return t("keyDetail.overview.spend.resetsAt", { time: resetTime });
    return hasOwnBudget ? t("keyDetail.overview.spend.neverResets") : t("keyDetail.overview.spend.neverResetsNoBudget");
  })();
  return (
    <div>
      <div className="text-xs font-medium text-muted-foreground">
        {resetTime ? t("keyDetail.overview.spend.thisPeriod") : t("keyDetail.overview.spend.spend")}
      </div>
      <p className="m-0 mt-1 text-2xl leading-8 font-semibold tabular-nums" data-testid="key-period-spend">
        {formatMoney(spend)}
      </p>
      <p className="m-0 text-xs text-muted-foreground" data-testid="key-spend-reset">
        {hint}
      </p>
    </div>
  );
}

function LifetimeStat({ totalSpend, lifetimeNeedsBackfill, createdAt }: SpendCardProps) {
  const { t } = useTranslation();
  const total = totalSpend ?? 0;
  return (
    <div data-testid="key-lifetime-spend">
      <div className="flex items-center gap-1 text-xs font-medium text-muted-foreground">
        {t("keyDetail.overview.spend.lifetime")}
        {lifetimeNeedsBackfill ? (
          <HoverCard>
            <HoverCardTrigger
              render={
                <button
                  type="button"
                  aria-label={t("keyDetail.keyInfoView.overview.lifetimeSpendHintLabel")}
                  className="inline-flex cursor-help text-muted-foreground"
                  data-testid="key-lifetime-spend-backfill-hint"
                />
              }
            >
              <Info className="size-3" />
            </HoverCardTrigger>
            <HoverCardContent className="w-80">
              {t("keyDetail.keyInfoView.overview.lifetimeSpendHint")}
            </HoverCardContent>
          </HoverCard>
        ) : null}
      </div>
      <p className="m-0 mt-1 text-[15px] leading-[22px] font-semibold tabular-nums">{formatMoney(total)}</p>
      {createdAt && !lifetimeNeedsBackfill ? (
        <p className="m-0 text-xs text-muted-foreground">
          {t("keyDetail.overview.spend.since", { date: formatDate(createdAt) })}
        </p>
      ) : null}
    </div>
  );
}

function UsedUpBadge({ throttles }: { throttles: boolean }) {
  const { t } = useTranslation();
  return throttles ? (
    <StatusBadge tone="warning" label={t("keyDetail.overview.budgets.throttled")} />
  ) : (
    <StatusBadge tone="error" label={t("keyDetail.overview.budgets.blocking")} />
  );
}

function CanStillSpendSlot({ budgets, keyThrottlesOnBudget, budgetResetAt }: SpendCardProps) {
  const { t } = useTranslation();
  const result = canStillSpend(budgets, keyThrottlesOnBudget);
  const label = (
    <div className="mb-2 text-xs font-medium text-muted-foreground">{t("keyDetail.overview.canStillSpend.title")}</div>
  );

  if (result.kind !== "measured") {
    return (
      <div className="min-w-0" data-testid="key-can-still-spend">
        {label}
        <div className="flex min-h-[76px] items-center rounded-md border border-dashed border-border px-3 py-2 text-muted-foreground">
          {result.kind === "noBudget"
            ? t("keyDetail.overview.canStillSpend.noBudget")
            : t("keyDetail.overview.canStillSpend.unmeasured")}
        </div>
      </div>
    );
  }

  const { budget, remaining, spend, throttles } = result;
  const usedUp = remaining === 0;
  const owner = ownerText(budget, usedUp);
  const reset = resetSentence(budget, formatBudgetResetTime(budgetResetAt));
  const figures = t("keyDetail.overview.canStillSpend.figures", {
    spend: formatMoney(spend),
    limit: formatMoney(budget.maxBudget),
  });
  return (
    <div className="min-w-0" data-testid="key-can-still-spend">
      {label}
      <div className="inline-flex items-center gap-1.5">
        <span className={cn("text-2xl leading-8 font-semibold tabular-nums", usedUp && "text-destructive")}>
          {formatMoney(remaining)}
        </span>
        {usedUp ? <UsedUpBadge throttles={throttles} /> : null}
      </div>
      <p className="m-0 mt-1 mb-2">
        {t(headroomKey(usedUp, throttles), { owner: t(owner.key, owner.values) })}
        <span className="text-muted-foreground">
          {usedUp
            ? t("keyDetail.overview.canStillSpend.usedUpFigures", { figures })
            : t("keyDetail.overview.canStillSpend.usedFigures", { figures })}
        </span>
      </p>
      <Meter
        value={Math.min(spend, budget.maxBudget)}
        max={budget.maxBudget}
        aria-label={t("keyDetail.overview.canStillSpend.title")}
      >
        <MeterTrack>
          <MeterIndicator tone={meterTone(budgetUsagePercent(budget) ?? 0)} />
        </MeterTrack>
      </Meter>
      <Caption className="mt-1">{t(reset.key, reset.values)}</Caption>
    </div>
  );
}

function BudgetRow({
  budget,
  keyThrottlesOnBudget,
  budgetResetAt,
  budgetHref,
}: SpendCardProps & { budget: AppliedBudget }) {
  const { t } = useTranslation();
  const usage = budgetUsagePercent(budget);
  const throttles = budgetThrottles(budget, keyThrottlesOnBudget);
  const scopeLabel = t(`keyDetail.overview.budgets.scopes.${SCOPE_KEYS[budget.scope]}` as ParseKeys);
  const caption = resetCaption(budget, formatBudgetResetTime(budgetResetAt));
  return (
    <li className="grid grid-cols-1 items-center gap-x-6 gap-y-1 border-t border-border py-2.5 md:grid-cols-[minmax(160px,260px)_minmax(120px,1fr)_minmax(180px,220px)]">
      <div className="min-w-0">
        <div className="flex min-w-0 items-center gap-2">
          <ResourceLink href={budget.scope === "Key" ? undefined : budgetHref(budget)}>{budget.name}</ResourceLink>
          {usage !== null && usage >= 100 ? <UsedUpBadge throttles={throttles} /> : null}
        </div>
        <Caption>
          {throttles ? t("keyDetail.overview.budgets.throttlesInstead", { scope: scopeLabel }) : scopeLabel}
        </Caption>
      </div>
      <div className="min-w-0">
        {usage === null ? (
          <Caption>{t("keyDetail.overview.budgets.spendNotReported")}</Caption>
        ) : (
          <Meter
            value={Math.min(usage, 100)}
            max={100}
            aria-label={scopeLabel}
            aria-valuetext={t("shared.appliedBudgets.percentUsed", { percent: Math.round(usage) })}
          >
            <MeterTrack>
              <MeterIndicator tone={meterTone(usage)} />
            </MeterTrack>
          </Meter>
        )}
      </div>
      <div className="tabular-nums md:text-right">
        <div>
          {budget.spend == null ? (
            <span className="font-medium">
              {t("keyDetail.overview.budgets.limitOnly", { limit: formatMoney(budget.maxBudget) })}
            </span>
          ) : (
            <>
              <span className="font-medium">{formatMoney(budget.spend)}</span>
              <span className="text-muted-foreground">
                {t("keyDetail.overview.budgets.ofLimit", { limit: formatMoney(budget.maxBudget) })}
              </span>
              <span className="inline-block min-w-11 text-right font-medium">
                {t("keyDetail.overview.budgets.percent", { percent: Math.round(usage ?? 0) })}
              </span>
            </>
          )}
        </div>
        <Caption>{t(caption.key, caption.values)}</Caption>
      </div>
    </li>
  );
}

function AlertsLine({ alerts }: { alerts: readonly BudgetAlertRow[] }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-start gap-2 text-muted-foreground">
      <Bell className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0">
        {t("keyDetail.overview.spend.alertsTitle")}
        <ul className="m-0 mt-1 flex list-none flex-col gap-0.5 p-0">
          {alerts.map((alert) => (
            <li key={alert.threshold ?? alert.emails}>
              {t("keyDetail.overview.spend.alertAt")}
              <span className="font-medium text-foreground tabular-nums">
                {t("keyDetail.overview.budgets.percent", { percent: alert.threshold })}
              </span>
              {t("keyDetail.overview.spend.alertEmails")}
              <span className="text-foreground">{alert.emails}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

function FallbacksLine({ fallbacks }: { fallbacks: Readonly<Record<string, readonly string[]>> }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-start gap-2 text-muted-foreground">
      <CornerDownRight className="mt-0.5 size-4 shrink-0" aria-hidden />
      <div className="min-w-0">
        {t("keyDetail.overview.spend.fallbacksTitle")}
        <ul className="m-0 mt-1 flex list-none flex-col gap-0.5 p-0">
          {Object.entries(fallbacks).map(([model, targets]) => (
            <li key={model} className="flex flex-wrap items-center gap-1.5">
              <span className="font-medium text-foreground">{model}</span>
              <ArrowRight className="size-3.5" aria-label={t("keyDetail.overview.spend.fallsBackTo")} />
              <span className="text-foreground">{targets.join(", ")}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
