import type { DateRangePickerValue } from "@/components/shared/date_picker_types";
import type { TFunction } from "i18next";
import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { toast } from "@/lib/toast";
import AdvancedDatePicker from "@/components/shared/advanced_date_picker";
import { BarChart } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { RefreshCw } from "lucide-react";
import { cachingHealthCheckCall } from "@/components/networking";
import { deriveErrorMessage } from "@/lib/http/client";
import { ErrorState } from "@/components/shared/ErrorState";
import { useCacheActivity, type CacheActivityGroup } from "@/app/(dashboard)/hooks/caching/useCacheActivity";

// Import the new component
import { CacheHealthTab } from "./cache_health";
import CacheSettings from "./cache_settings";
import CoordinationRedisSettings from "./coordination_redis_settings";
import { ErrorDrilldownCard } from "./ErrorDrilldown";

const REQUEST_SERIES = {
  apiRequests: "caching.dashboard.series.apiRequests",
  cacheHits: "caching.dashboard.series.cacheHits",
  failed: "caching.dashboard.series.failed",
} as const;

const UNKNOWN_CALL_TYPE = "Unknown";

const UNKNOWN_CALL_TYPE_NOTE = "caching.dashboard.requestsChart.unknownCallTypeNote";

const toChartDatum = (group: CacheActivityGroup, t: TFunction) => ({
  name: group.call_type,
  [t(REQUEST_SERIES.apiRequests)]: group.api_requests,
  [t(REQUEST_SERIES.cacheHits)]: group.cache_hits,
  [t(REQUEST_SERIES.failed)]: group.failed_requests,
  [t("caching.dashboard.series.cachedCompletionTokens")]: group.cached_completion_tokens,
  [t("caching.dashboard.series.generatedCompletionTokens")]: group.generated_completion_tokens,
});

const formatDateWithoutTZ = (date: Date | undefined) => {
  if (!date) return undefined;
  return date.toISOString().split("T")[0];
};

const resolveDrilldownCallType = (selected: string | null, groups: readonly CacheActivityGroup[]): string | null =>
  selected !== null && groups.some((group) => group.call_type === selected && group.failed_requests > 0)
    ? selected
    : null;

function valueFormatterNumbers(number: number) {
  const formatter = new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
    notation: "compact",
    compactDisplay: "short",
  });

  return formatter.format(number);
}

interface CachePageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  premiumUser: boolean;
}

// Helper function to deep-parse a JSON string if possible

const CacheDashboard: React.FC<CachePageProps> = ({ accessToken, token, userRole, userID, premiumUser }) => {
  const { t } = useTranslation();
  const anchor1 = useComboboxAnchor();
  const anchor2 = useComboboxAnchor();
  const [selectedApiKeys, setSelectedApiKeys] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [errorDrilldownCallType, setErrorDrilldownCallType] = useState<string | null>(null);

  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [lastRefreshed, setLastRefreshed] = useState("");
  const [healthCheckResponse, setHealthCheckResponse] = useState<any>("");

  const {
    data: activity,
    isError: activityFailed,
    error: activityError,
    refetch,
  } = useCacheActivity({
    startDate: formatDateWithoutTZ(dateValue.from),
    endDate: formatDateWithoutTZ(dateValue.to),
    keyAliases: selectedApiKeys,
    models: selectedModels,
  });

  useEffect(() => {
    setLastRefreshed(new Date().toLocaleString());
  }, []);

  const uniqueApiKeys = activity?.filter_options.key_aliases ?? [];
  const uniqueModels = activity?.filter_options.models ?? [];
  const chartData = (activity?.groups ?? []).map((group) => toChartDatum(group, t));
  const hasUnknownGroup = (activity?.groups ?? []).some((group) => group.call_type === UNKNOWN_CALL_TYPE);
  const activeDrilldownCallType = resolveDrilldownCallType(errorDrilldownCallType, activity?.groups ?? []);

  const handleRefreshClick = () => {
    refetch();
    setLastRefreshed(new Date().toLocaleString());
  };

  const runCachingHealthCheck = async () => {
    try {
      toast.info(t("caching.dashboard.healthCheck.running"));
      setHealthCheckResponse("");
      const response = await cachingHealthCheckCall(accessToken !== null ? accessToken : "");
      setHealthCheckResponse(response);
    } catch (error: any) {
      console.error("Error running health check:", error);
      let errorData;
      if (error && error.message) {
        try {
          // Parse the error message which may contain a nested error layer.
          let parsedData = JSON.parse(error.message);
          // If the parsed object is wrapped (e.g. { error: { ... } }), unwrap it.
          if (parsedData.error) {
            parsedData = parsedData.error;
          }
          errorData = parsedData;
        } catch (e) {
          errorData = { message: error.message };
        }
      } else {
        errorData = { message: t("caching.dashboard.healthCheck.unknownError") };
      }
      setHealthCheckResponse({ error: errorData });
    }
  };

  const totals = activity?.totals;
  const hasRequests = totals != null && totals.api_requests + totals.cache_hits + totals.failed_requests > 0;
  const statCards = [
    {
      label: t("caching.dashboard.stats.cacheHitRatio"),
      value: `${hasRequests ? totals.cache_hit_ratio.toFixed(2) : "0"}%`,
    },
    { label: t("caching.dashboard.stats.cacheHits"), value: valueFormatterNumbers(totals?.cache_hits ?? 0) },
    {
      label: t("caching.dashboard.stats.cachedCompletionTokens"),
      value: valueFormatterNumbers(totals?.cached_completion_tokens ?? 0),
    },
  ];

  return (
    <Tabs defaultValue="analytics" className="mt-2 mb-8 w-full gap-2 p-8">
      <div className="mt-2 flex w-full items-center justify-between border-b">
        <TabsList variant="line" className="h-auto rounded-none p-0">
          <TabsTrigger value="analytics" className="flex-none rounded-none px-4 py-2">
            {t("caching.dashboard.tabs.analytics")}
          </TabsTrigger>
          <TabsTrigger value="health" className="flex-none rounded-none px-4 py-2">
            {t("caching.dashboard.tabs.health")}
          </TabsTrigger>
          <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
            {t("caching.dashboard.tabs.settings")}
          </TabsTrigger>
          <TabsTrigger value="coordination" className="flex-none rounded-none px-4 py-2">
            {t("caching.dashboard.tabs.coordination")}
          </TabsTrigger>
        </TabsList>

        <div className="flex items-center space-x-2">
          {lastRefreshed && (
            <p className="text-sm text-muted-foreground">
              {t("caching.dashboard.lastRefreshed")}
              {lastRefreshed}
            </p>
          )}
          <Button
            variant="outline"
            size="icon-sm"
            onClick={handleRefreshClick}
            aria-label={t("caching.dashboard.refresh")}
          >
            <RefreshCw />
          </Button>
        </div>
      </div>

      <TabsContent value="analytics" keepMounted>
        <Card>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              <Trans
                i18nKey="caching.dashboard.description"
                components={{
                  cacheDocs: (
                    <a
                      href="https://docs.litellm.ai/docs/proxy/caching"
                      target="_blank"
                      rel="noreferrer"
                      className="underline"
                    />
                  ),
                  promptDocs: (
                    <a
                      href="https://docs.litellm.ai/docs/completion/prompt_caching"
                      target="_blank"
                      rel="noreferrer"
                      className="underline"
                    />
                  ),
                }}
              />
            </p>

            <div className="mt-4 grid grid-cols-1 items-center gap-4 md:grid-cols-[1fr_1fr_auto]">
              <Combobox
                multiple
                items={uniqueApiKeys}
                value={selectedApiKeys}
                onValueChange={(keys: string[]) => setSelectedApiKeys(keys)}
              >
                <ComboboxChips render={<div ref={anchor1} />}>
                  <ComboboxValue>
                    {(keys: string[]) =>
                      keys.map((key) => (
                        <ComboboxChip key={key} aria-label={key}>
                          {key}
                        </ComboboxChip>
                      ))
                    }
                  </ComboboxValue>
                  <ComboboxChipsInput placeholder={t("caching.dashboard.filters.virtualKeysPlaceholder")} />
                </ComboboxChips>
                <ComboboxContent anchor={anchor1}>
                  <ComboboxEmpty>{t("caching.dashboard.filters.noVirtualKeys")}</ComboboxEmpty>
                  <ComboboxList>
                    {(key: string) => (
                      <ComboboxItem key={key} value={key}>
                        {key}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>

              <Combobox
                multiple
                items={uniqueModels}
                value={selectedModels}
                onValueChange={(models: string[]) => setSelectedModels(models)}
              >
                <ComboboxChips render={<div ref={anchor2} />}>
                  <ComboboxValue>
                    {(models: string[]) =>
                      models.map((model) => (
                        <ComboboxChip key={model} aria-label={model}>
                          {model}
                        </ComboboxChip>
                      ))
                    }
                  </ComboboxValue>
                  <ComboboxChipsInput placeholder={t("caching.dashboard.filters.modelsPlaceholder")} />
                </ComboboxChips>
                <ComboboxContent anchor={anchor2}>
                  <ComboboxEmpty>{t("caching.dashboard.filters.noModels")}</ComboboxEmpty>
                  <ComboboxList>
                    {(model: string) => (
                      <ComboboxItem key={model} value={model}>
                        {model}
                      </ComboboxItem>
                    )}
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>

              <AdvancedDatePicker
                value={dateValue}
                onValueChange={(value) => {
                  setDateValue(value);
                }}
              />
            </div>

            <div className="mt-4 grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
              {statCards.map((stat) => (
                <Card key={stat.label}>
                  <CardContent>
                    <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
                    <div className="mt-2 flex items-baseline space-x-2.5">
                      <p className="text-3xl font-semibold">{stat.value}</p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            <Card className="mt-4">
              <CardHeader>
                <CardTitle className="text-base font-semibold">{t("caching.dashboard.requestsChart.title")}</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">{t("caching.dashboard.requestsChart.hint")}</p>
                {hasUnknownGroup && <p className="mt-1 text-sm text-muted-foreground">{t(UNKNOWN_CALL_TYPE_NOTE)}</p>}
                {activityFailed ? (
                  <ErrorState
                    title={t("caching.dashboard.requestsChart.loadFailed")}
                    message={deriveErrorMessage(activityError)}
                    onRetry={() => void refetch()}
                  />
                ) : (
                  <BarChart
                    data={chartData}
                    stack={true}
                    index="name"
                    valueFormatter={valueFormatterNumbers}
                    categories={[t(REQUEST_SERIES.apiRequests), t(REQUEST_SERIES.cacheHits), t(REQUEST_SERIES.failed)]}
                    colors={["sky", "teal", "red"]}
                    yAxisWidth={48}
                    className="mt-2"
                    onValueChange={(item) => {
                      if (item.categoryClicked === t(REQUEST_SERIES.failed)) setErrorDrilldownCallType(item.name);
                    }}
                  />
                )}
              </CardContent>
            </Card>

            {activeDrilldownCallType !== null && (
              <ErrorDrilldownCard
                callType={activeDrilldownCallType}
                buckets={activity?.error_breakdown ?? []}
                valueFormatter={valueFormatterNumbers}
                onClose={() => setErrorDrilldownCallType(null)}
              />
            )}

            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="text-base font-semibold">{t("caching.dashboard.tokensChart.title")}</CardTitle>
              </CardHeader>
              <CardContent>
                <BarChart
                  data={chartData}
                  stack={true}
                  index="name"
                  valueFormatter={valueFormatterNumbers}
                  categories={[
                    t("caching.dashboard.series.generatedCompletionTokens"),
                    t("caching.dashboard.series.cachedCompletionTokens"),
                  ]}
                  colors={["sky", "teal"]}
                  yAxisWidth={48}
                />
              </CardContent>
            </Card>
          </CardContent>
        </Card>
      </TabsContent>

      <TabsContent value="health" keepMounted>
        <CacheHealthTab
          accessToken={accessToken}
          healthCheckResponse={healthCheckResponse}
          runCachingHealthCheck={runCachingHealthCheck}
        />
      </TabsContent>

      <TabsContent value="settings" keepMounted>
        <CacheSettings accessToken={accessToken} userRole={userRole} userID={userID} />
      </TabsContent>

      <TabsContent value="coordination" keepMounted>
        <CoordinationRedisSettings />
      </TabsContent>
    </Tabs>
  );
};

export default CacheDashboard;
