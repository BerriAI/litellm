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
const auth = vi.hoisted(() => ({ premiumUser: false }));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "proxy_admin", premiumUser: auth.premiumUser }),
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

const OFF = {
  ...SETTINGS,
  groups: SETTINGS.groups.map((info) => ({ ...info, enabled: false })),
} as TelemetrySettings;

const renderPage = () =>
  render(
    <QueryClientProvider client={new QueryClient()}>
      <NuqsTestingAdapter hasMemory>
        <TelemetrySettingsPage />
      </NuqsTestingAdapter>
    </QueryClientProvider>,
  );

describe("TelemetrySettingsPage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    auth.premiumUser = false;
  });

  it("says nothing is recorded without a destination and gates picking one behind the enterprise modal", async () => {
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

    expect(await screen.findByRole("dialog", { name: "Telemetry is an enterprise-only feature" })).toBeVisible();
    expect(put).not.toHaveBeenCalled();
  });

  it.each([
    [false, "Telemetry is an enterprise-only feature"],
    [true, "Contact your enterprise representative"],
  ])("never turns telemetry on from the top switch (enterprise: %s)", async (premiumUser, title) => {
    auth.premiumUser = premiumUser;
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: OFF, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    renderPage();

    const telemetry = await screen.findByRole("switch", { name: "Telemetry" });
    expect(telemetry).not.toBeChecked();
    expect(screen.getByRole("switch", { name: "Heartbeat" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("radio", { name: /^Local table/ })).toHaveAttribute("aria-disabled", "true");
    await user.click(telemetry);

    expect(await screen.findByRole("dialog", { name: title })).toBeVisible();
    expect(put).not.toHaveBeenCalled();
    expect(telemetry).not.toBeChecked();
  });

  it("turns every group off from the top switch and keeps the saved destination", async () => {
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: SETTINGS, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole("switch", { name: "Telemetry" }));

    expect(put).toHaveBeenCalledWith("/telemetry/settings", { body: { groups: [], destination: "local_table" } });
  });

  it("gates turning another group on behind the enterprise modal while telemetry is on", async () => {
    vi.spyOn(backgroundFetchClient, "GET").mockResolvedValue({ data: SETTINGS, response: new Response() } as never);
    const put = vi.spyOn(fetchClient, "PUT").mockReturnValue(new Promise(() => undefined) as never);
    const user = userEvent.setup();
    renderPage();

    await user.click(await screen.findByRole("tab", { name: "Admin UI" }));
    await user.click(await screen.findByRole("switch", { name: "Page navigation" }));

    expect(await screen.findByRole("dialog", { name: "Telemetry is an enterprise-only feature" })).toBeVisible();
    expect(put).not.toHaveBeenCalled();
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
