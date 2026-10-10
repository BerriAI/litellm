import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { backgroundFetchClient, fetchClient } from "@/lib/http/api";
import type { TelemetrySettings } from "@/app/(dashboard)/hooks/telemetry/useTelemetrySettings";
import TelemetrySettingsPage from "./TelemetrySettingsPage";

vi.mock("@/lib/telemetry/uiEvents", async (original) => ({
  ...(await original<typeof import("@/lib/telemetry/uiEvents")>()),
  recordUiEvent: vi.fn(() => Promise.resolve()),
}));
vi.mock("next/navigation", () => ({ usePathname: () => "/ui/telemetry" }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "proxy_admin" }),
}));

const SETTINGS = {
  destination: "local_table",
  editable: true,
  environment_variables: [],
  flush_interval_seconds: 300,
  report: null,
  report_is_sample: true,
  retention_days: 30,
  groups: [
    { group: "heartbeat", enabled: true, requires: null },
    { group: "request_success", enabled: true, requires: "heartbeat" },
    { group: "page_navigation", enabled: false, requires: "heartbeat" },
  ],
} as unknown as TelemetrySettings;

describe("TelemetrySettingsPage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("locks the other tab's switches while a save from one tab is still in flight", async () => {
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: SETTINGS, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={new QueryClient()}>
        <NuqsTestingAdapter hasMemory>
          <TelemetrySettingsPage />
        </NuqsTestingAdapter>
      </QueryClientProvider>,
    );

    await user.click(await screen.findByRole("switch", { name: "Heartbeat" }));
    await user.click(screen.getByRole("tab", { name: "Admin UI" }));

    expect(await screen.findByRole("switch", { name: "Page navigation" })).toHaveAttribute("aria-disabled", "true");
    expect(put).toHaveBeenCalledTimes(1);
  });
});
