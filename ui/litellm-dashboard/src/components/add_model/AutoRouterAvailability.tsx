import { useLanguage } from "@/contexts/LanguageContext";
import { createContext, useContext, useEffect, useState } from "react";
import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { apiClient } from "@/components/networking";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import type { components } from "@/lib/http/schema";
import { englishTranslate, type Translate } from "@/i18n/language";

type Availability = components["schemas"]["AutoRouterAvailabilityResponse"];
type Request = components["schemas"]["AutoRouterAvailabilityRequest"];
export type Allowance = components["schemas"]["AutoRouterAllowance"];

type AvailabilityState = {
  data?: Availability;
  isPending: boolean;
  isError: boolean;
  isChecking?: boolean;
  refetch?: () => unknown;
};

export const AutoRouterAvailabilityContext = createContext<AvailabilityState>({ isPending: true, isError: false });

export const useAutoRouterAvailability = (accessToken: string, body: Request, enabled = true) => {
  const { t } = useLanguage();
  const serialized = JSON.stringify(body.complexity_router_config ?? null);
  const [debounced, setDebounced] = useState(serialized);
  useEffect(() => {
    const timeout = setTimeout(() => setDebounced(serialized), 300);
    return () => clearTimeout(timeout);
  }, [serialized]);
  const options: UseQueryOptions<Availability> = {
    queryKey: ["autoRouterAvailability", accessToken, body.team_id, body.saved_model_id, debounced],
    queryFn: ({ signal }) =>
      apiClient.post<Availability>("/auto_router/availability", {
        accessToken,
        body: { ...body, complexity_router_config: JSON.parse(debounced) },
        signal,
      }),
    enabled: enabled && Boolean(accessToken),
    placeholderData: (previous, previousQuery) => {
      const key = previousQuery?.queryKey;
      return key?.[1] === accessToken && key[2] === body.team_id && key[3] === body.saved_model_id
        ? previous
        : undefined;
    },
    refetchOnMount: "always",
    staleTime: 0,
    retry: false,
  };
  const query = useQuery(options);
  const isChecking = query.isFetching || query.isPlaceholderData || serialized !== debounced;
  const saveBlockedReason = () => {
    if (!enabled) return null;
    if (query.isPending || isChecking) return t("Checking availability");
    if (query.isError || !query.data) return t("Could not check availability. Retry before saving.");
    return query.data.error ?? null;
  };
  return {
    ...query,
    isPending: query.isPending || (query.isFetching && !query.isFetchedAfterMount),
    isChecking,
    saveBlockedReason: saveBlockedReason(),
  };
};

export const allowanceLabel = (allowance?: Allowance, t: Translate = englishTranslate): string | null => {
  if (!allowance?.available) return t("Availability unavailable");
  if (allowance.limit == null) return null;
  if (allowance.used_by_this_router) return t("Used by this router");
  return t("{remaining} of {limit} available", { remaining: String(allowance.remaining), limit: allowance.limit });
};

const availabilityLabel = (state: AvailabilityState, key: string, t: Translate) => {
  if (state.isPending || state.isChecking) return t("Checking availability");
  if (state.isError) return t("Availability unavailable");
  return allowanceLabel(
    state.data?.allowances.find((entry) => entry.key === key),
    t,
  );
};

export const useAllowanceLabel = (key: string) => {
  const { t } = useLanguage();
  return availabilityLabel(useContext(AutoRouterAvailabilityContext), key, t);
};

export const isAllowanceExhausted = (allowance?: Allowance) =>
  Boolean(allowance?.available && allowance.limit != null && allowance.remaining === 0) &&
  !allowance?.used_by_this_router;

export const AUTO_ROUTER_CONTACT_URL = "https://calendly.com/tin-berri/litellm-auto-router-pricing-discussion";

export const AutoRouterContactLink = ({ features, message }: { features?: string[]; message?: string }) => {
  const { t } = useLanguage();
  const state = useContext(AutoRouterAvailabilityContext);
  if (state.isPending || state.isError || state.isChecking) return null;
  const exhausted = state.data?.allowances.some(
    (entry) => (!features || features.includes(entry.key)) && isAllowanceExhausted(entry),
  );
  if (!exhausted) return null;
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-1 text-xs leading-5 text-muted-foreground">
      {message}
      <a
        href={AUTO_ROUTER_CONTACT_URL}
        target="_blank"
        rel="noopener noreferrer"
        className="font-medium text-blue-600 hover:underline dark:text-blue-400"
      >
        {t("Talk to our team")}
      </a>
    </span>
  );
};

export const AutoRouterAllowanceLabel = ({ feature }: { feature: string }) => {
  const label = useAllowanceLabel(feature);
  return label ? (
    <span className="shrink-0 whitespace-nowrap text-xs leading-5 tabular-nums text-muted-foreground">{label}</span>
  ) : null;
};

export const AutoRouterAllowanceNote = ({ feature, label }: { feature: string; label: string }) => {
  const availability = useAllowanceLabel(feature);
  return availability ? (
    <p className="text-xs leading-5 text-muted-foreground">
      {label}: {availability} <AutoRouterContactLink features={[feature]} />
    </p>
  ) : null;
};

export const AutoRouterLimits = () => {
  const { t } = useLanguage();
  const state = useContext(AutoRouterAvailabilityContext);
  const limits = [
    ["heuristic_v2", "Heuristic v2 routers"],
    ["capability", "Capability routers"],
    ["llm_v2", "Fuse v2 routers"],
    ["tier_or_classifier_prompt", "Custom tiers or prompts"],
    ["heuristic_tuning", "Rule-based tuning"],
  ];
  return (
    <Popover>
      <PopoverTrigger className="shrink-0 whitespace-nowrap text-xs font-normal text-muted-foreground underline underline-offset-4 hover:text-foreground">
        {t("View limits")}
      </PopoverTrigger>
      <PopoverContent align="end" className="w-96 max-w-[calc(100vw-2rem)] gap-3">
        <PopoverTitle>{t("Routing and customization limits")}</PopoverTitle>
        <p className="text-xs leading-5 text-muted-foreground">
          {t(
            "Rule-based, Complexity, and OSS are unlimited with built-in settings. Choose or change tier models freely. Customization allowances are shared across this proxy.",
          )}
        </p>
        <dl className="space-y-2 text-xs">
          {limits.map(([key, label]) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <dt>{t(label)}</dt>
              <dd className="shrink-0 tabular-nums text-muted-foreground">
                {availabilityLabel(state, key, t) ?? t("Unlimited")}
              </dd>
            </div>
          ))}
        </dl>
        <p className="text-xs leading-5 text-muted-foreground">
          {t(
            "Custom tier definitions and written classifier instructions share one allowance. Built-in prompts and display-name changes do not use it.",
          )}
        </p>
        <p className="text-xs leading-5 text-muted-foreground">
          {t(
            "Changing scoring rules, such as weights, thresholds, keywords, or custom dimensions, uses the Rule-based tuning allowance. It also applies to Heuristic first and Hybrid. Recorded settings on existing routers are preserved; new routers start from built-in rules.",
          )}
        </p>
        <AutoRouterContactLink message={t("Need a higher limit?")} />
      </PopoverContent>
    </Popover>
  );
};
