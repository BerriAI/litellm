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
  stored_destination: "local_table",
  destination_set_by_environment: false,
  available_destinations: ["local_table"],
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

const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <NuqsTestingAdapter hasMemory>
        <TelemetrySettingsPage />
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );

describe("TelemetrySettingsPage", () => {
  afterEach(() => vi.restoreAllMocks());

  it("says nothing is recorded until a destination is picked, then saves the picked one with the current groups", async () => {
    const unchosen = { ...SETTINGS, destination: null, stored_destination: null };
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: unchosen, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    renderPage();

    expect(
      await screen.findByText("Telemetry is on but no destination is chosen, so nothing is recorded"),
    ).toBeVisible();
    expect(screen.getByRole("radio", { name: /^HTTP/ })).toHaveAttribute("aria-disabled", "true");
    await user.click(screen.getByRole("radio", { name: /^Local table/ }));

    expect(put).toHaveBeenCalledWith("/telemetry/settings", {
      body: { groups: ["heartbeat", "request_success"], destination: "local_table" },
    });
  });

  it("keeps the saved destination when a group is switched", async () => {
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: SETTINGS, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole("switch", { name: "Request success" }));

    expect(put).toHaveBeenCalledWith("/telemetry/settings", {
      body: { groups: ["heartbeat"], destination: "local_table" },
    });
  });

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
