"use client";

import type { ParseKeys } from "i18next";
import { Ban, CircleAlert } from "lucide-react";
import { useTranslation } from "react-i18next";

import { Button } from "@/components/ui/button";
import { Alert, AlertAction, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import type { AppliedBudget } from "@/components/shared/appliedBudgets";
import { mapInternalToDisplayNames } from "@/components/callback_info_helpers";
import {
  extractLoggingSettings,
  filterSensitiveMetadata,
  formatMetadataForDisplay,
  stripTagsFromMetadata,
} from "@/components/key_info_utils";
import {
  budgetAlertRowsFromMetadata,
  KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
} from "@/components/key_team_helpers/budgetAlertEmails";
import type { KeyResponse } from "@/components/key_team_helpers/key_list";
import { formatBudgetResetTime } from "@/utils/budgetUtils";

import { AccessCard } from "./AccessCard";
import { ownerText, resetSentence, SCOPE_KEYS } from "./budgetText";
import { usedUpBudget } from "./canStillSpend";
import { formatMoney } from "./formatters";
import { GuardrailsCard } from "./GuardrailsCard";
import { hasRateLimits, keyAccess, keyRateLimits, stringList, unsetSections } from "./keyOverviewSections";
import { LifecycleCard } from "./LifecycleCard";
import { LoggingCard, type LoggingIntegration } from "./LoggingCard";
import { OtherSettings, type OtherSettingsData } from "./OtherSettings";
import { RateLimitsCard } from "./RateLimitsCard";
import { SpendCard } from "./SpendCard";

export interface KeyOverviewProps {
  keyData: KeyResponse;
  budgets: readonly AppliedBudget[];
  budgetHref: (budget: AppliedBudget) => string | undefined;
  team: { id: string; name: string } | null;
  createdBy: { label: string; userId: string } | null;
  accessGroupNames: ReadonlySet<string> | undefined;
  policyGuardrails: Readonly<Record<string, readonly string[]>>;
  loadingPolicies: boolean;
  lifetimeNeedsBackfill: boolean;
  onOpenActivity: () => void;
  onUnblock?: () => void;
  nowMs: number;
}

const isLoggingIntegration = (value: unknown): value is LoggingIntegration =>
  typeof value === "object" &&
  value !== null &&
  typeof (value as Record<string, unknown>).callback_name === "string" &&
  typeof (value as Record<string, unknown>).callback_type === "string";

const numberOrNull = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const numberPairs = (value: unknown): readonly (readonly [string, number])[] =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? Object.entries(value).flatMap(([name, tokens]) => (typeof tokens === "number" ? [[name, tokens] as const] : []))
    : [];

const otherSettingsData = (key: KeyResponse): OtherSettingsData => {
  const metadata = key.metadata ?? {};
  const visibleMetadata = filterSensitiveMetadata(stripTagsFromMetadata(metadata));
  return {
    tags: stringList(metadata.tags),
    allowedRoutes: stringList(key.allowed_routes),
    passThroughRoutes: stringList(metadata.allowed_passthrough_routes),
    prompts: stringList(metadata.prompts),
    promptCaching: Boolean(metadata.enable_prompt_caching),
    estimatedOutputTokens: numberOrNull(metadata.default_estimated_output_tokens),
    estimatedOutputTokensPerModel: numberPairs(metadata.default_estimated_output_tokens_per_model),
    routerSettings: key.router_settings,
    metadataJson: formatMetadataForDisplay(stripTagsFromMetadata(metadata)),
    metadataFieldCount: Object.keys(visibleMetadata).length,
  };
};

export function KeyOverview(props: KeyOverviewProps) {
  const { t } = useTranslation();
  const { keyData, budgets } = props;
  const metadata = keyData.metadata ?? {};
  const keyThrottlesOnBudget = metadata.throttle_on_budget_exceeded === true;
  const rateLimits = keyRateLimits(keyData);
  const access = keyAccess(keyData.object_permission);
  const guardrails = stringList(metadata.guardrails);
  const policies = stringList(metadata.policies);
  const integrations = extractLoggingSettings(metadata).filter(isLoggingIntegration);
  const disabledCallbacks = mapInternalToDisplayNames([...stringList(metadata.litellm_disabled_callbacks)]);
  const configured = { rateLimits, access, guardrails, policies, loggingCount: integrations.length, disabledCallbacks };
  const unset = unsetSections(configured);
  const usedUp = usedUpBudget(budgets, keyThrottlesOnBudget);
  const showGuardrails = guardrails.length > 0 || policies.length > 0;
  const showLogging = integrations.length > 0 || disabledCallbacks.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {keyData.blocked ? (
        <Alert variant="warning">
          <Ban aria-hidden />
          <AlertTitle>{t("keyDetail.overview.alerts.blockedTitle")}</AlertTitle>
          <AlertDescription>{t("keyDetail.overview.alerts.blockedDescription")}</AlertDescription>
          {props.onUnblock ? (
            <AlertAction>
              <Button variant="outline" size="sm" onClick={props.onUnblock}>
                {t("keyDetail.overview.alerts.unblock")}
              </Button>
            </AlertAction>
          ) : null}
        </Alert>
      ) : null}
      {usedUp ? (
        <UsedUpAlert budget={usedUp} budgetResetAt={keyData.budget_reset_at} href={props.budgetHref(usedUp)} />
      ) : null}
      <SpendCard
        spend={keyData.spend}
        totalSpend={keyData.total_spend}
        lifetimeNeedsBackfill={props.lifetimeNeedsBackfill}
        budgetDuration={keyData.budget_duration || null}
        budgetResetAt={keyData.budget_reset_at || null}
        createdAt={keyData.created_at || null}
        hasOwnBudget={keyData.max_budget != null}
        budgets={budgets}
        keyThrottlesOnBudget={keyThrottlesOnBudget}
        alerts={budgetAlertRowsFromMetadata(metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY)}
        fallbacks={keyData.budget_fallbacks ?? {}}
        budgetHref={props.budgetHref}
        onOpenActivity={props.onOpenActivity}
      />
      <LifecycleCard source={keyData} createdBy={props.createdBy} nowMs={props.nowMs} />
      {hasRateLimits(rateLimits) ? (
        <RateLimitsCard limits={rateLimits} accessGroupNames={props.accessGroupNames} />
      ) : null}
      <AccessCard
        models={stringList(keyData.models)}
        team={props.team}
        access={access}
        accessGroupNames={props.accessGroupNames}
      />
      {showGuardrails || showLogging ? (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(360px,100%),1fr))] items-start gap-4">
          {showGuardrails ? (
            <GuardrailsCard
              guardrails={guardrails}
              policies={policies}
              policyGuardrails={props.policyGuardrails}
              loadingPolicies={props.loadingPolicies}
              globalGuardrailsOff={metadata.disable_global_guardrails === true}
            />
          ) : null}
          {showLogging ? <LoggingCard integrations={integrations} disabledCallbacks={disabledCallbacks} /> : null}
        </div>
      ) : null}
      {unset.length > 0 ? (
        <div
          className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded-xl border border-dashed border-border px-4 py-3 text-muted-foreground"
          data-testid="key-not-set"
        >
          <span className="text-xs font-medium">{t("keyDetail.overview.notSet.title")}</span>
          <span>
            {unset.map((section) => t(`keyDetail.overview.notSet.sections.${section}` as ParseKeys)).join(", ")}
          </span>
        </div>
      ) : null}
      <OtherSettings data={otherSettingsData(keyData)} accessGroupNames={props.accessGroupNames} />
    </div>
  );
}

function UsedUpAlert({
  budget,
  budgetResetAt,
  href,
}: {
  budget: AppliedBudget;
  budgetResetAt: string | null;
  href: string | undefined;
}) {
  const { t } = useTranslation();
  const scope = SCOPE_KEYS[budget.scope];
  const reset = resetSentence(budget, formatBudgetResetTime(budgetResetAt));
  const owner = ownerText(budget, true);
  return (
    <Alert variant="error">
      <CircleAlert aria-hidden />
      <AlertTitle>{t("keyDetail.overview.alerts.usedUpTitle", { owner: t(owner.key, owner.values) })}</AlertTitle>
      <AlertDescription>
        {t("keyDetail.overview.alerts.usedUpDescription", {
          spend: formatMoney(budget.spend ?? 0),
          limit: formatMoney(budget.maxBudget),
          reset: t(reset.key, reset.values),
        })}
      </AlertDescription>
      {href && budget.scope !== "Key" ? (
        <AlertAction>
          <Button variant="outline" size="sm" render={<a href={href} />} nativeButton={false}>
            {t(`keyDetail.overview.alerts.open.${scope}` as ParseKeys)}
          </Button>
        </AlertAction>
      ) : null}
    </Alert>
  );
}
