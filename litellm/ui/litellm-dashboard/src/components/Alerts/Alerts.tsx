"use client";

import { parseAsBoolean, parseAsString, parseAsStringEnum, parseAsStringLiteral, useQueryStates } from "nuqs";
import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Page, PageTabs, PageTabsList, PageTabsTrigger } from "@/components/shared/Page";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { isProxyAdminTierRole } from "@/utils/roles";
import { useDeprecationAlerts } from "@/components/ModelDeprecationAlerts/useDeprecationAlerts";
import ModelDeprecationsPanel from "@/components/ModelDeprecationAlerts/ModelDeprecationsPanel";
import DeprecationEmailSetup from "@/components/ModelDeprecationAlerts/DeprecationEmailSetup";
import { AlertStatus, effectiveAlertStatus } from "@/components/ModelDeprecationAlerts/migrationState";
import { useAlert, useAlerts } from "./useAlerts";
import { ALERT_TYPES, AlertFilter, AlertItem, TYPE_LABEL_KEYS, presentAlert } from "./alertPresentation";
import AlertActions from "./AlertActions";
import AlertChart from "./AlertChart";
import AlertDetail from "./AlertDetail";
import ManageAlerts from "./ManageAlerts";

const parsers = {
  alert: parseAsString,
  model: parseAsString,
  migration: parseAsBoolean.withDefault(false),
  state: parseAsStringEnum(Object.values(AlertStatus)).withDefault(AlertStatus.Open),
  type: parseAsStringLiteral(ALERT_TYPES).withDefault("all"),
};
function LoadError({ children, retry }: { children: React.ReactNode; retry: () => void }) {
  const { t } = useTranslation();
  return (
    <div role="alert" className="flex items-center gap-3 py-4 text-sm">
      <p>{children}</p>
      <Button variant="outline" size="sm" onClick={retry}>
        {t("alerts.panel.retry")}
      </Button>
    </div>
  );
}
export default function Alerts() {
  const { t } = useTranslation();
  const auth = useAuthorized();
  const [params, setParams] = useQueryStates(parsers);
  const allowed = Boolean(auth.isAuthorized && isProxyAdminTierRole(auth.userRole));
  if (auth.isLoading)
    return (
      <Page>
        <p role="status">{t("alerts.unified.loading")}</p>
      </Page>
    );
  if (!allowed)
    return (
      <Page>
        <p>{t("alerts.unified.adminRequired")}</p>
      </Page>
    );
  return (
    <Page className="mx-auto max-w-[1440px]">
      {params.alert || params.model ? (
        <SelectedAlert
          auth={auth}
          onBack={() => void setParams({ alert: null, model: null, migration: null }, { history: "push" })}
        />
      ) : (
        <AlertsInbox auth={auth} />
      )}
    </Page>
  );
}

function SelectedAlert({ auth, onBack }: { auth: ReturnType<typeof useAuthorized>; onBack: () => void }) {
  const { t } = useTranslation();
  const [params] = useQueryStates(parsers);
  const scope = JSON.stringify([auth.userId, auth.userRole]);
  const detail = useAlert(params.alert, scope, true);
  const deprecations = useDeprecationAlerts(params.model ? auth.accessToken : null);
  const selectedDeprecation = deprecations.data?.find((alert) => alert.usageModel === params.model);
  const selected: AlertItem | undefined = params.alert
    ? detail.data && { source: "performance", alert: detail.data }
    : selectedDeprecation && { source: "deprecation", alert: selectedDeprecation };
  if (params.migration && params.model) return <ModelDeprecationsPanel />;
  const query = params.alert ? detail : deprecations;
  if (selected && !query.isError)
    return <AlertDetail item={selected} readOnly={Boolean(auth.isViewOnly)} onBack={onBack} />;
  return (
    <>
      <Button variant="ghost" className="w-fit" onClick={onBack}>
        {t("alerts.unified.back")}
      </Button>
      {query.isError ? (
        <LoadError retry={() => void query.refetch()}>{t("alerts.unified.loadError")}</LoadError>
      ) : (
        <p role="status">{query.isPending ? t("alerts.panel.loadingAlert") : t("alerts.unified.unavailable")}</p>
      )}
    </>
  );
}

function AlertsInbox({ auth }: { auth: ReturnType<typeof useAuthorized> }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [params, setParams] = useQueryStates(parsers);
  const scope = JSON.stringify([auth.userId, auth.userRole]);
  const showDeprecations = params.type === "all" || params.type === "model-deprecations";
  const showPerformance = params.type !== "model-deprecations";
  const list = useAlerts(scope, showPerformance, params.state, params.type);
  const deprecations = useDeprecationAlerts(showDeprecations ? auth.accessToken : null);
  const retirementItems: AlertItem[] = (deprecations.data ?? [])
    .filter((alert) => effectiveAlertStatus(alert) === params.state)
    .map((alert) => ({ source: "deprecation", alert }));
  const performanceItems: AlertItem[] = (list.data?.pages.flatMap((page) => page.data) ?? []).map((alert) => ({
    source: "performance",
    alert,
  }));
  const items = [...retirementItems, ...performanceItems];
  const loading = list.isLoading || deprecations.isLoading;
  const hasError = (showPerformance && list.isError) || (showDeprecations && deprecations.isError);
  function refresh() {
    void client.invalidateQueries({ queryKey: ["alert-monitoring"] });
    if (showPerformance) void list.refetch();
    if (showDeprecations) void deprecations.refetch();
  }
  return (
    <>
      <ManageAlerts scope={scope} readOnly={Boolean(auth.isViewOnly)} />
      <PageTabs
        value={params.state}
        onValueChange={(value) => void setParams({ state: value as AlertStatus })}
        className="flex-none"
      >
        <PageTabsList>
          {Object.values(AlertStatus).map((status) => (
            <PageTabsTrigger key={status} value={status} className="capitalize">
              {t(`alerts.status.${status}`)}
            </PageTabsTrigger>
          ))}
        </PageTabsList>
      </PageTabs>
      <div className="flex items-center justify-between gap-3">
        <Select value={params.type} onValueChange={(value) => void setParams({ type: value as AlertFilter })}>
          <SelectTrigger className="w-52" aria-label={t("alerts.unified.typeFilter")}>
            <SelectValue>{t(TYPE_LABEL_KEYS[params.type])}</SelectValue>
          </SelectTrigger>
          <SelectContent>
            {ALERT_TYPES.map((type) => (
              <SelectItem key={type} value={type}>
                {t(TYPE_LABEL_KEYS[type])}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button variant="outline" size="sm" disabled={list.isFetching || deprecations.isFetching} onClick={refresh}>
          <RefreshCw className="size-3.5" />
          {t("alerts.unified.refresh")}
        </Button>
      </div>
      {showDeprecations && deprecations.isError && (
        <LoadError retry={() => void deprecations.refetch()}>{t("alerts.unified.deprecationsError")}</LoadError>
      )}
      {showPerformance && list.isError && (
        <LoadError retry={() => void list.refetch()}>{t("alerts.unified.performanceError")}</LoadError>
      )}
      <InboxNotice loading={loading} error={hasError} count={items.length} status={params.state} />
      <ul className="divide-y divide-border">
        {items.map((item) => {
          const alert = presentAlert(item);
          return (
            <li
              key={`${item.source}-${alert.id}`}
              className="grid min-w-0 gap-6 py-6 lg:grid-cols-[minmax(0,1fr)_200px]"
            >
              <div className="min-w-0 space-y-3">
                <button
                  className="text-left text-base font-semibold leading-6 hover:underline break-words"
                  onClick={() =>
                    void setParams(
                      item.source === "performance"
                        ? { alert: item.alert.id, model: null }
                        : { alert: null, model: item.alert.usageModel },
                      { history: "push" },
                    )
                  }
                >
                  {alert.title}
                </button>
                <p className="text-sm leading-6 text-muted-foreground">{alert.description}</p>
                <AlertActions item={item} readOnly={Boolean(auth.isViewOnly)} />
              </div>
              <div className="hidden self-center lg:block">
                <AlertChart alert={alert} compact />
              </div>
            </li>
          );
        })}
      </ul>
      {showPerformance && list.hasNextPage && (
        <Button
          variant="outline"
          className="self-center"
          disabled={list.isFetchingNextPage}
          onClick={() => void list.fetchNextPage()}
        >
          {list.isFetchingNextPage ? t("alerts.unified.loadingMore") : t("alerts.unified.loadMore")}
        </Button>
      )}
      <DeprecationEmailSetup />
    </>
  );
}

function InboxNotice({
  loading,
  error,
  count,
  status,
}: {
  loading: boolean;
  error: boolean;
  count: number;
  status: AlertStatus;
}) {
  const { t } = useTranslation();
  if (count > 0 || error) return null;
  return (
    <p role={loading ? "status" : undefined} className="py-8 text-sm text-muted-foreground">
      {loading ? t("alerts.unified.loading") : t(`alerts.unified.empty.${status}`)}
    </p>
  );
}
