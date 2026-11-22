import { useTranslation } from "react-i18next";
import type { ReactNode } from "react";
import { formatCompact } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { TableCell, TableHead } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import type { CustomerUsageEntity } from "./customerUsageApi";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { customerBudgetPeriod, customerBudgetStatus, customerBudgetUsage } from "./customerBudgetUsage";
import { CustomerBudgetLimitCell } from "./CustomerBudgetColumn";

export function CustomerSpendTier({ entity }: { entity: CustomerUsageEntity }) {
  const { t } = useTranslation();
  const custom = customerBudgetUsage(entity).length > 0 || entity.rateLimits;
  const label = entity.budgetGroup ?? (custom ? t("customerPortal.organizationGroups.customBadge") : "—");
  return (
    <TableCell className="px-4 text-muted-foreground">
      <span className="block max-w-48 truncate" title={label}>
        {label}
      </span>
    </TableCell>
  );
}

export function CustomerSpendIdentity({
  entity: member,
  onClick,
  href,
  usageLayout = false,
  nameClassName,
  idClassName,
}: {
  entity: CustomerUsageEntity;
  usageLayout?: boolean;
  nameClassName?: string;
  idClassName?: string;
} & ({ href: string; onClick?: never } | { href?: never; onClick: () => void })) {
  const Name = href ? "a" : "button";
  return (
    <div className="flex items-center gap-3">
      <span
        aria-hidden="true"
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium",
          usageLayout && "size-9",
        )}
      >
        {member.label
          .split(/\s+/)
          .slice(0, 2)
          .map((part) => part.charAt(0))
          .join("")
          .toLocaleUpperCase()}
      </span>
      <div className="min-w-0">
        <Name
          type={href ? undefined : "button"}
          href={href}
          title={member.label}
          className={cn(
            "block max-w-52 truncate text-left font-medium hover:text-primary focus-visible:outline-2 focus-visible:outline-ring",
            nameClassName,
          )}
          onClick={onClick}
        >
          {member.label}
        </Name>
        {member.label !== member.id && (
          <span
            className={cn("block max-w-52 truncate font-mono text-[11px] text-muted-foreground", idClassName)}
            title={member.id}
          >
            {member.id}
          </span>
        )}
      </div>
    </div>
  );
}

export function CustomerSpendHeaders({
  children,
  usageLayout = false,
  showYearly = true,
  shareLabel,
}: {
  children: ReactNode;
  usageLayout?: boolean;
  showYearly?: boolean;
  shareLabel?: string;
}) {
  const { t } = useTranslation();
  return (
    <>
      <TableHead className={cn("min-w-56", usageLayout && "w-[30%] min-w-80 px-4")}>
        {t(usageLayout ? "customerPortal.list.columns.monthlySpend" : "customerPortal.analytics.monthlySpend")}
      </TableHead>
      {children}
      {showYearly && (
        <TableHead className={cn("text-right", usageLayout && "px-4")}>
          {t("customerPortal.analytics.yearlySpend")}
        </TableHead>
      )}
      <CustomerSpendMetricsHeaders usageLayout={usageLayout} shareLabel={shareLabel} />
      <TableHead className={cn("text-right", usageLayout && "px-4")}>{t("customerPortal.analytics.status")}</TableHead>
    </>
  );
}

export function CustomerSpendCells({
  entity: member,
  monthlySpend,
  yearlySpend,
  tokens,
  organizationSpend,
  usageLayout = false,
  tokenContent,
  shareContent,
  showYearly = true,
  shareLabel,
  shareSpend = monthlySpend,
  budgetUnavailable,
  spendClassName,
  selectedSpend,
}: {
  entity: CustomerUsageEntity;
  monthlySpend: number | null;
  yearlySpend: number | null;
  tokens: number | null;
  organizationSpend: number | null;
  usageLayout?: boolean;
  tokenContent?: ReactNode;
  shareContent?: ReactNode;
  showYearly?: boolean;
  shareLabel?: string;
  shareSpend?: number | null;
  budgetUnavailable?: ReactNode;
  spendClassName?: string;
  selectedSpend?: number | null;
}) {
  const { t } = useTranslation();
  const budgets = budgetUnavailable ? [] : customerBudgetUsage(member);
  const monthlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "monthly");
  const yearlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "yearly");
  const otherBudgets = budgets.filter(
    (budget) => budget.source === "window" && customerBudgetPeriod(budget.budget_duration) === "other",
  );
  const customWindows = otherBudgets.length > 0;

  const status = customerBudgetStatus(budgets);
  const atLimit = status === "customerPortal.analytics.reachedLimit";

  return (
    <>
      <TableCell className={cn("pr-4", usageLayout && "px-4")}>
        <div className={cn("space-y-3", spendClassName)}>
          <SelectedPeriodSpend spend={selectedSpend} hasMonthlyBudget={monthlyBudgets.length > 0} />
          {monthlyBudgets.map((budget) => (
            <CustomerBudgetWindow
              key={`${budget.source}-${budget.budget_duration}`}
              budget={budget}
              showReset={monthlyBudgets.length > 1}
            />
          ))}
          {monthlyBudgets.length === 0 && selectedSpend === undefined && (
            <CustomerBudgetSpend
              spend={monthlySpend}
              limit={null}
              limitKnown={!budgetUnavailable}
              label={t("customerPortal.analytics.monthlySpend")}
              showLimit={!customWindows}
              noLimitLabel={t("customerPortal.analytics.noMonthlyLimit")}
            />
          )}
          {otherBudgets.map((budget) => (
            <div key={`${budget.source}-${budget.budget_duration}`} className="space-y-1 text-xs text-muted-foreground">
              <span>
                {t("customerPortal.analytics.budgetPeriod")}
                {budget.budget_duration && ` (${budget.budget_duration})`}
              </span>
              <CustomerBudgetWindow budget={budget} />
            </div>
          ))}
        </div>
      </TableCell>
      {budgetUnavailable ? (
        <TableCell className="px-4">{budgetUnavailable}</TableCell>
      ) : (
        <CustomerBudgetLimitCell entity={member} />
      )}
      {showYearly && (
        <TableCell className={cn("text-right tabular-nums", usageLayout && "px-4")}>
          <div className="space-y-2">
            {yearlyBudgets.map((budget) => (
              <CustomerBudgetWindow
                key={`${budget.source}-${budget.budget_duration}`}
                budget={budget}
                showReset={yearlyBudgets.length > 1}
                stacked={usageLayout}
              />
            ))}
            {yearlyBudgets.length === 0 && (
              <CustomerBudgetSpend
                spend={yearlySpend}
                limit={null}
                limitKnown={!budgetUnavailable}
                label={t("customerPortal.analytics.yearlySpend")}
                showLimit={!customWindows}
                noLimitLabel={t("customerPortal.analytics.noYearlyLimit")}
              />
            )}
          </div>
        </TableCell>
      )}
      <CustomerSpendMetricsCells
        monthlySpend={shareSpend}
        organizationSpend={organizationSpend}
        usageLayout={usageLayout}
        shareContent={shareContent}
        shareLabel={shareLabel}
        tokenContent={tokenContent ?? (tokens === null ? "—" : formatCompact(tokens))}
      />
      <TableCell className={cn("text-right", usageLayout && "px-4")}>
        <span
          className={cn(
            "inline-flex rounded-full bg-muted px-2 py-0.5 text-xs",
            usageLayout && "whitespace-nowrap",
            atLimit && "bg-destructive/10 text-destructive",
          )}
        >
          {t(budgetUnavailable ? "customerPortal.analytics.budgetUsageUnavailable" : status)}
        </span>
      </TableCell>
    </>
  );
}

function SelectedPeriodSpend({ spend, hasMonthlyBudget }: { spend?: number | null; hasMonthlyBudget: boolean }) {
  const { t } = useTranslation();
  if (spend === undefined) return null;
  return (
    <>
      <CustomerBudgetSpend spend={spend} limit={null} label={t("customerPortal.analytics.spend")} showLimit={false} />
      {hasMonthlyBudget && (
        <span className="text-xs text-muted-foreground">{t("customerPortal.analytics.monthlySpend")}</span>
      )}
    </>
  );
}

export function CustomerSpendMetricsHeaders({
  usageLayout = false,
  shareLabel,
}: {
  usageLayout?: boolean;
  shareLabel?: string;
}) {
  const { t } = useTranslation();
  return (
    <>
      <TableHead className={cn("min-w-36 whitespace-normal text-right", usageLayout && "px-4 text-left")}>
        {shareLabel ?? t("customerPortal.analytics.customerSpendShare")}
      </TableHead>
      <TableHead className={cn("text-right", usageLayout && "px-4")}>{t("customerPortal.analytics.tokens")}</TableHead>
    </>
  );
}

export function CustomerSpendMetricsCells({
  monthlySpend,
  organizationSpend,
  usageLayout = false,
  tokenContent,
  shareContent,
  shareLabel,
}: {
  monthlySpend: number | null;
  organizationSpend: number | null;
  usageLayout?: boolean;
  tokenContent: ReactNode;
  shareContent?: ReactNode;
  shareLabel?: string;
}) {
  return (
    <>
      <CustomerSpendShare
        monthlySpend={monthlySpend}
        organizationSpend={organizationSpend}
        usageLayout={usageLayout}
        label={shareLabel}
      >
        {shareContent}
      </CustomerSpendShare>
      <TableCell className={cn("text-right tabular-nums", usageLayout && "px-4")}>{tokenContent}</TableCell>
    </>
  );
}

function CustomerSpendShare({
  monthlySpend,
  organizationSpend,
  usageLayout,
  children,
  label,
}: {
  monthlySpend: number | null;
  organizationSpend: number | null;
  usageLayout: boolean;
  children?: ReactNode;
  label?: string;
}) {
  const { t } = useTranslation();
  const spendShare = organizationSpend && monthlySpend !== null ? (monthlySpend / organizationSpend) * 100 : 0;
  const shareAvailable = organizationSpend !== null && monthlySpend !== null;
  const sharePercentage = Math.max(0, Math.min(100, spendShare));
  const shareLabel = t("format.number", {
    val: spendShare / 100,
    style: "percent",
    maximumFractionDigits: 1,
  });
  return (
    <TableCell className={usageLayout ? "px-4" : undefined}>
      {children ??
        (shareAvailable ? (
          <div className={usageLayout ? "flex items-center gap-3" : undefined}>
            <div
              role="meter"
              aria-label={label ?? t("customerPortal.analytics.customerSpendShare")}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={sharePercentage}
              aria-valuetext={shareLabel}
              title={shareLabel}
              className={cn(
                "ml-auto h-2.5 w-24 overflow-hidden rounded-full bg-muted",
                usageLayout && "ml-0 h-2 w-14 shrink-0",
              )}
            >
              <div
                className="h-full rounded-full bg-primary"
                style={{ width: `${sharePercentage}%`, opacity: usageLayout ? 0.6 : 0.3 + sharePercentage * 0.007 }}
              />
            </div>
            {usageLayout && <span className="text-xs tabular-nums text-muted-foreground">{shareLabel}</span>}
          </div>
        ) : (
          <span className="text-muted-foreground">—</span>
        ))}
    </TableCell>
  );
}
