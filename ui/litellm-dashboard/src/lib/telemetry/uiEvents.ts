import { queryClient } from "@/contexts/ReactQueryProvider";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type UiEvent = components["schemas"]["UIEventBody"];

export const TELEMETRY_WINDOW_MS = 300_000;

export const pageNavigationQuery = {
  queryKey: ["telemetry", "pageNavigationEnabled"],
  queryFn: async (): Promise<boolean> => {
    const { data } = await backgroundFetchClient.GET("/telemetry/ui_events/enabled");
    return data?.enabled ?? false;
  },
  staleTime: TELEMETRY_WINDOW_MS,
  retry: false,
} as const;

export const recordUiEvent = (event: UiEvent): Promise<void> =>
  queryClient
    .fetchQuery(pageNavigationQuery)
    .then((enabled) => (enabled ? backgroundFetchClient.POST("/telemetry/ui_events", { body: event }) : undefined))
    .then(() => undefined)
    .catch(() => undefined);
