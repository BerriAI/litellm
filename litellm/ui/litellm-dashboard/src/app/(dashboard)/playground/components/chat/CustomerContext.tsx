"use client";

import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { extractProxyErrorMessage } from "@/lib/http/client";
import type { ResolvedCustomerHierarchy } from "../../../customers/_components/customerHierarchyApi";
import { TargetPicker } from "./Pickers";
import { remainingCustomerBudget, safeCustomerValue, type CustomerRequestFailure } from "./customerContext";
import type { usePlaygroundCustomer } from "./usePlaygroundCustomer";

type CustomerState = ReturnType<typeof usePlaygroundCustomer>;

export function CustomerControl({ customer }: { customer: CustomerState }) {
  const { t } = useTranslation();
  const organizations = customer.organizations.data ?? [];
  const error = customer.linkError ?? customer.organizations.error ?? customer.members.error;
  return (
    <Popover>
      <PopoverTrigger render={<Button variant="ghost" size="sm" className="h-7 max-w-52 truncate text-xs" />}>
        <span className="truncate">
          {t("playground.chat.customer.title")}
          {customer.alias ? ` · ${customer.alias}` : ""}
        </span>
      </PopoverTrigger>
      <PopoverContent side="top" align="start" className="w-96 space-y-3">
        <p className="text-sm font-medium">{t("playground.chat.customer.title")}</p>
        <TargetPicker
          label={t("playground.chat.customer.organization")}
          placeholder={t("playground.chat.customer.selectOrg")}
          groups={[
            {
              label: t("playground.chat.customer.organization"),
              options: organizations
                .filter((org) => org.alias && safeCustomerValue(org.alias))
                .map((org) => ({ value: org.customer_org_id, label: org.alias!, description: org.customer_org_id })),
            },
          ]}
          value={customer.organization?.customer_org_id ?? null}
          onValueChange={(id) =>
            customer.setContext(organizations.find((org) => org.customer_org_id === id) ?? null, null)
          }
          isLoading={customer.organizations.isLoading}
          emptyText={t("playground.chat.customer.aliasRequired")}
        />
        {customer.organization && (
          <>
            <TargetPicker
              label={t("playground.chat.customer.member")}
              placeholder={t("playground.chat.customer.optionalMember")}
              groups={[
                {
                  label: t("playground.chat.customer.member"),
                  options: (customer.members.data ?? []).map((member) => ({
                    value: member.customer_member_id,
                    label: member.alias || member.customer_member_id,
                    description: `${t("playground.chat.customer.team")} · ${member.parent_team_id}`,
                  })),
                },
              ]}
              value={customer.memberId}
              onValueChange={(id) => customer.setContext(customer.organization!, id)}
              isLoading={customer.members.isLoading}
              emptyText={t("playground.chat.customer.noMembers")}
            />
            <Button variant="ghost" size="sm" onClick={() => customer.setContext(customer.organization!, null)}>
              {t("playground.chat.customer.orgOnly")}
            </Button>
          </>
        )}
        <p className="text-xs text-muted-foreground">{t("playground.chat.customer.headersHint")}</p>
        {error && (
          <p role="alert" className="text-xs text-destructive">
            {extractProxyErrorMessage(error)}
          </p>
        )}
        <Button variant="outline" size="sm" onClick={() => customer.setContext(null, null)}>
          {t("playground.chat.customer.clear")}
        </Button>
      </PopoverContent>
    </Popover>
  );
}

const money = (value: number): string =>
  new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 6,
  }).format(value);

export function CustomerBudgetStrip({
  context,
  error,
  loading,
  unavailable,
}: {
  context?: ResolvedCustomerHierarchy;
  error: CustomerRequestFailure | null;
  loading: boolean;
  unavailable: boolean;
}) {
  const { t } = useTranslation();
  const scopes = context
    ? [
        {
          label: t("playground.chat.customer.organization"),
          id: context.organization.alias || context.organization.customer_org_id,
          spend: context.organization.spend,
          max: context.organization.budget?.max_budget,
        },
        ...(context.member
          ? [
              {
                label: t("playground.chat.customer.member"),
                id: context.member.alias || context.member.customer_member_id,
                spend: context.member.spend,
                max: context.member.effective_budget.max_budget,
              },
            ]
          : []),
      ]
    : [];
  return (
    <section
      aria-label={t("playground.chat.customer.budgets")}
      aria-live="polite"
      className="mx-4 mt-2 space-y-1 rounded-md border bg-muted/30 px-3 py-2 text-xs"
    >
      <div className="flex flex-wrap gap-x-5 gap-y-1">
        {scopes.map((scope) => {
          const remaining = remainingCustomerBudget(scope.spend, scope.max);
          return (
            <p key={scope.label}>
              <span className="font-medium">
                {scope.label} · {scope.id}
              </span>{" "}
              {t("playground.chat.customer.spend", { value: money(scope.spend) })}
              {" · "}
              {t("playground.chat.customer.remaining", {
                value: remaining === null ? t("playground.chat.customer.unlimited") : money(remaining),
              })}
            </p>
          );
        })}
      </div>
      <p className="text-muted-foreground">
        {loading ? t("playground.chat.customer.refreshing") : t("playground.chat.customer.settled")}
      </p>
      {unavailable && (
        <p role="alert" className="text-destructive">
          {t("playground.chat.customer.unavailable")}
        </p>
      )}
      {error && (
        <p role="alert" className="break-words font-medium text-destructive">
          {error.status !== null ? `HTTP ${error.status} · ` : ""}
          {error.message}
        </p>
      )}
    </section>
  );
}
