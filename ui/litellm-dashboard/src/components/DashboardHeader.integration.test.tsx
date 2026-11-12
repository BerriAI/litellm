import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PluginModeProvider } from "@/contexts/PluginModeContext";
import { CommandPaletteProvider } from "@/components/CommandPalette/CommandPaletteProvider";
import { DashboardHeader } from "./DashboardHeader";

vi.mock("next/navigation", () => ({
  usePathname: () => "/ui/logs",
  useRouter: () => ({ push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

let role = "proxy_admin";
const alertRequests: string[] = [];
const performanceAlert = {
  id: "9007199254740993",
  type: "spend",
  resource_type: "team",
  resource_id: "example-team",
  resource_name: "Research",
  status: "open",
  window_start: "2026-11-10T00:00:00Z",
  window_end: "2026-11-11T00:00:00Z",
  created_at: "2026-11-11T03:00:00Z",
  updated_at: "2026-11-11T03:00:00Z",
  evidence: { value: 1200, baseline: 200, unit: "usd", sample_size: 1000 },
};
let openAlerts = [performanceAlert];
let rejectAlerts = false;
let rejectDeprecations = false;

beforeEach(() => {
  role = "proxy_admin";
  alertRequests.length = 0;
  openAlerts = [performanceAlert];
  rejectAlerts = false;
  rejectDeprecations = false;
  localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url.endsWith("/management/v1/users/me")) {
        return Response.json({ user_id: "example-admin", user_role: role });
      }
      if (url.endsWith("/model/deprecations?warn_within_days=30")) {
        if (rejectDeprecations) return Response.json({ detail: "Unavailable" }, { status: 503 });
        return Response.json({ deprecated: [], imminent: [] });
      }
      if (url.includes("/alerts?")) {
        alertRequests.push(url);
        if (rejectAlerts) return Response.json({ detail: "Unavailable" }, { status: 503 });
        return Response.json({ data: openAlerts, next_cursor: "next-page" });
      }
      if (url.endsWith("/get/ui_settings")) return Response.json({ values: {} });
      if (url.endsWith("/litellm/.well-known/litellm-ui-config")) {
        const config = {
          server_root_path: "",
          proxy_base_url: "",
          is_control_plane: true,
          workers: [
            { worker_id: "primary", name: "Primary worker", url: "http://primary.test" },
            { worker_id: "backup", name: "Backup worker", url: "http://backup.test" },
          ],
        };
        return Response.json(config);
      }
      if (url.endsWith("/api/plugins")) {
        return Response.json([{ name: "metrics", display_name: "Metrics", url: "/metrics" }]);
      }
      if (url.endsWith("/public/litellm_blog_posts")) {
        return Response.json({
          posts: [
            {
              title: "Gateway update",
              description: "Release notes",
              date: "2026-01-01",
              url: "https://example.com/update",
            },
          ],
        });
      }
      return Response.json({}, { status: 404 });
    }),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

function renderHeader(accessToken: string | null = null) {
  const user = userEvent.setup();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <PluginModeProvider accessToken="test-session">
        <CommandPaletteProvider>
          <DashboardHeader accessToken={accessToken} />
        </CommandPaletteProvider>
      </PluginModeProvider>
    </QueryClientProvider>,
  );
  return { user, client };
}

describe("dashboard header controls", () => {
  it("opens search from both header controls", async () => {
    const { user } = renderHeader();
    await user.click(screen.getByRole("button", { name: "Search or jump to…" }));
    expect(await screen.findByRole("dialog", { name: "Command palette" })).toBeVisible();
    await user.keyboard("{Escape}");
    await user.click(screen.getByRole("button", { name: "Search", exact: true }));
    expect(await screen.findByRole("dialog", { name: "Command palette" })).toBeVisible();
  });

  it("offers docs, community and latest posts under the bell", async () => {
    const { user } = renderHeader();
    await user.click(screen.getByRole("button", { name: /Notifications/ }));
    const tools = await screen.findByRole("dialog", { name: "Notifications" });
    expect(await within(tools).findByRole("link", { name: /Gateway update/ })).toHaveAttribute(
      "href",
      "https://example.com/update",
    );
    expect(within(tools).getByRole("link", { name: "Docs", exact: true })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/",
    );
    expect(within(tools).getByRole("link", { name: "Discord", exact: true })).toBeVisible();
  });

  it("respects the saved blog and community visibility settings", async () => {
    localStorage.setItem("disableBlogPosts", "true");
    localStorage.setItem("disableShowPrompts", "true");
    const { user } = renderHeader();
    expect(screen.queryByRole("link", { name: "Join Slack" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Notifications/ }));
    const tools = await screen.findByRole("dialog", { name: "Notifications" });
    expect(within(tools).getByRole("link", { name: "Docs", exact: true })).toBeVisible();
    expect(within(tools).queryByRole("link", { name: "Blog", exact: true })).not.toBeInTheDocument();
    expect(within(tools).queryByRole("link", { name: "Discord", exact: true })).not.toBeInTheDocument();
  });

  it("allows filtering and selecting a worker", async () => {
    localStorage.setItem("litellm_selected_worker_id", "primary");
    const { user } = renderHeader();
    const worker = await screen.findByRole("combobox", { name: "Worker" });
    await user.clear(worker);
    await user.type(worker, "Backup");
    await user.click(await screen.findByRole("option", { name: "Backup worker" }));
    expect(localStorage.getItem("litellm_selected_worker_id")).toBeNull();
  });
});

describe("header performance notifications", () => {
  it.each(["proxy_admin", "proxy_admin_viewer"])(
    "loads open alerts for %s and refreshes after resolution",
    async (userRole) => {
      role = userRole;
      const { user, client } = renderHeader("example-session");
      await user.click(await screen.findByRole("button", { name: "Notifications, alerts need attention" }));
      expect(screen.getByRole("link", { name: /Spend on Research/ })).toHaveAttribute(
        "href",
        `/ui/alerts?alert=${performanceAlert.id}`,
      );
      expect(alertRequests).toHaveLength(1);
      const query = new URL(alertRequests[0]).searchParams;
      expect(query.get("status")).toBe("open");
      expect(query.get("type")).toBe("performance");
      openAlerts = [];
      await client.invalidateQueries({ queryKey: ["alerts"] });
      expect(await screen.findByText("No open alerts.")).toBeVisible();
      expect(screen.queryByRole("link", { name: /Spend on Research/ })).not.toBeInTheDocument();
      client.clear();
    },
  );

  it("hides cached performance alerts when refreshing fails", async () => {
    const { user, client } = renderHeader("example-session");
    await user.click(await screen.findByRole("button", { name: "Notifications, alerts need attention" }));
    expect(screen.getByRole("link", { name: /Spend on Research/ })).toHaveTextContent("$1,200.00 vs. $200.00 usual");
    rejectAlerts = true;
    await client.invalidateQueries({ queryKey: ["alerts"] });
    expect(await screen.findByText("Could not load performance alerts.")).toBeVisible();
    expect(screen.queryByRole("link", { name: /Spend on Research/ })).not.toBeInTheDocument();
    expect(screen.queryByText("No open alerts.")).not.toBeInTheDocument();
    client.clear();
  });

  it.each([false, true])("reports deprecation failures with performance failure %s", async (performanceFails) => {
    openAlerts = [];
    rejectAlerts = performanceFails;
    const { user, client } = renderHeader("example-session");
    await user.click(screen.getByRole("button", { name: /Notifications/ }));
    if (performanceFails) {
      expect(await screen.findByText("Could not load performance alerts.")).toBeVisible();
    } else {
      expect(await screen.findByText("No open alerts.")).toBeVisible();
    }
    rejectDeprecations = true;
    await client.invalidateQueries({ queryKey: ["deprecationAlerts"] });
    expect(await screen.findByText("Could not load model deprecations.")).toBeVisible();
    expect(screen.queryByText("No open alerts.")).not.toBeInTheDocument();
    client.clear();
  });

  it.each(["internal_user", "org_admin"])("does not request performance alerts for %s", async (userRole) => {
    role = userRole;
    const { user, client } = renderHeader("example-session");
    await user.click(screen.getByRole("button", { name: /Notifications/ }));
    expect(await screen.findByRole("link", { name: /Gateway update/ })).toBeVisible();
    expect(alertRequests).toHaveLength(0);
    expect(screen.queryByRole("link", { name: /Spend on Research/ })).not.toBeInTheDocument();
    client.clear();
  });
});
