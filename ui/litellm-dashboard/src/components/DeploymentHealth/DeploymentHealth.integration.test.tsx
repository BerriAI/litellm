import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import type { DeploymentWarning } from "@/app/(dashboard)/hooks/deploymentWarnings/useDeploymentWarnings";
import { renderWithProviders, screen, testQueryClient, waitFor, within } from "../../../tests/test-utils";
import { createDeploymentHealthHandle, DeploymentHealthCriticalRow, DeploymentHealthIcon } from "./DeploymentHealth";
import { DeploymentHealthPanel } from "./DeploymentHealthPanel";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");

const NOW = new Date("2026-10-14T12:00:00Z");

interface PatchCall {
  id: string;
  body: { ignored: boolean; snooze_seconds?: number };
}

interface FakeProxy {
  warnings: DeploymentWarning[];
  ignoreAvailable: boolean;
  gets: number;
  patches: PatchCall[];
}

const proxy: FakeProxy = { warnings: [], ignoreAvailable: true, gets: 0, patches: [] };

const ENV_CREDENTIAL: DeploymentWarning = { id: "env_credential_login", severity: "critical", ignore: null };
const LICENSE_IN_3_DAYS: DeploymentWarning = { id: "license_expiry:2026-10-17", severity: "critical", ignore: null };
const NO_REDIS: DeploymentWarning = { id: "no_redis", severity: "warning", ignore: null };
const DEBUG: DeploymentWarning = { id: "detailed_debug", severity: "warning", ignore: null };

const snoozedBy = (user: string, until: string): DeploymentWarning["ignore"] => ({
  source: "admin",
  ignored_by: user,
  ignored_at: "2026-09-30T09:30:00Z",
  ignored_until: until,
});

const signIn = (userRole: string) => {
  const claims = { key: "sk-health-test", user_id: "admin-1", user_role: userRole, exp: NOW.getTime() / 1000 + 3600 };
  const token = `${btoa(JSON.stringify({ alg: "none" }))}.${btoa(JSON.stringify(claims))}.test`;
  document.cookie = `token=${token}; Path=/`;
};

const handleRequest = async (request: Request): Promise<Response> => {
  const { pathname } = new URL(request.url);
  if (pathname === "/litellm/.well-known/litellm-ui-config") {
    return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
  }
  if (pathname === "/health/deployment_warnings" && request.method === "GET") {
    proxy.gets += 1;
    return Response.json({ warnings: proxy.warnings, ignore_available: proxy.ignoreAvailable });
  }
  const patched = pathname.match(/^\/health\/deployment_warnings\/(.+)$/);
  if (patched && request.method === "PATCH") {
    const id = decodeURIComponent(patched[1]);
    const body: PatchCall["body"] = await request.json();
    proxy.patches.push({ id, body });
    const ignore = body.ignored
      ? {
          source: "admin" as const,
          ignored_by: "admin-1",
          ignored_at: NOW.toISOString(),
          ignored_until: body.snooze_seconds
            ? new Date(NOW.getTime() + body.snooze_seconds * 1000).toISOString()
            : null,
        }
      : null;
    proxy.warnings = proxy.warnings.map((warning) => (warning.id === id ? { ...warning, ignore } : warning));
    return Response.json({ id, ignore });
  }
  throw new Error(`Unexpected request: ${request.method} ${request.url}`);
};

const Harness = () => {
  const [handle] = useState(createDeploymentHealthHandle);
  return (
    <>
      <header>
        <DeploymentHealthIcon handle={handle} />
      </header>
      <DeploymentHealthCriticalRow handle={handle} />
      <DeploymentHealthPanel handle={handle} />
    </>
  );
};

const renderHealth = (userRole = "proxy_admin") => {
  signIn(userRole);
  return renderWithProviders(<Harness />);
};

const openPanel = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("button", { name: /^Deployment health/ }));
  return screen.findByRole("dialog", { name: "Deployment health" });
};

const EMPTY_PROXY: FakeProxy = { warnings: [], ignoreAvailable: true, gets: 0, patches: [] };

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW);
  testQueryClient.clear();
  Object.assign(proxy, EMPTY_PROXY, { patches: [] });
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL, init?: RequestInit) =>
      handleRequest(input instanceof Request ? input : new Request(new URL(String(input), "http://localhost"), init)),
    ),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.cookie = "token=; Max-Age=0; Path=/";
});

describe("deployment health icon", () => {
  it("should show a plain shield with no count when every warning is ignored", async () => {
    proxy.warnings = [{ ...NO_REDIS, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") }];
    renderHealth();

    await waitFor(() => expect(proxy.gets).toBe(1));
    const icon = await screen.findByRole("button", { name: "Deployment health" });
    expect(icon).toHaveTextContent(/^$/);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it.each([
    {
      name: "destructive when a critical warning is active",
      warnings: [ENV_CREDENTIAL, NO_REDIS],
      tone: "error",
    },
    { name: "quiet when only non-critical warnings are active", warnings: [NO_REDIS, DEBUG], tone: "neutral" },
  ])("should count active warnings and be $name", async ({ warnings, tone }) => {
    proxy.warnings = [...warnings, { ...LICENSE_IN_3_DAYS, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") }];
    renderHealth();

    const icon = await screen.findByRole("button", { name: "Deployment health, 2 active warnings" });
    expect(within(icon).getByText("2")).toHaveAttribute("data-tone", tone);
  });

  it("should not render or call the proxy for users below proxy admin viewer", async () => {
    proxy.warnings = [ENV_CREDENTIAL];
    renderHealth("internal_user");

    await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalled());
    expect(screen.queryByRole("button", { name: /Deployment health/ })).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(proxy.gets).toBe(0);
  });
});

describe("critical row", () => {
  it("should show the top critical warning and count the other critical ones", async () => {
    proxy.warnings = [LICENSE_IN_3_DAYS, ENV_CREDENTIAL, NO_REDIS];
    renderHealth();

    const row = await screen.findByRole("alert");
    expect(row).toHaveTextContent("Your LiteLLM Enterprise license expires in 3 days (Oct 17, 2026)");
    expect(row).toHaveTextContent("+1 more");
    expect(row).not.toHaveTextContent("Environment-credential login");
  });

  it("should not show a row when only non-critical warnings are active", async () => {
    proxy.warnings = [NO_REDIS, { ...ENV_CREDENTIAL, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") }];
    renderHealth();

    await screen.findByRole("button", { name: "Deployment health, 1 active warning" });
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("should open the panel from Review", async () => {
    const user = userEvent.setup();
    proxy.warnings = [ENV_CREDENTIAL];
    renderHealth();

    await user.click(await within(await screen.findByRole("alert")).findByRole("button", { name: "Review" }));

    const panel = await screen.findByRole("dialog", { name: "Deployment health" });
    expect(within(panel).getByRole("article", { name: "Environment-credential login is enabled" })).toBeVisible();
  });
});

describe("deployment health panel", () => {
  it("should group active warnings by severity and keep ignored ones collapsed with who and until when", async () => {
    const user = userEvent.setup();
    proxy.warnings = [
      ENV_CREDENTIAL,
      NO_REDIS,
      { ...DEBUG, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") },
      { id: "license_expiry:2026-10-01", severity: "critical", ignore: snoozedBy("admin-3", "2026-10-15T00:00:00Z") },
    ];
    renderHealth();
    const panel = await openPanel(user);

    expect(
      within(panel)
        .getAllByRole("region")
        .map((region) => region.getAttribute("aria-label")),
    ).toEqual(["Critical", "Warnings"]);
    const critical = within(panel).getByRole("region", { name: "Critical" });
    expect(within(critical).getByRole("article")).toHaveTextContent(
      "general_settings.disable_env_credential_login: true",
    );
    const warnings = within(panel).getByRole("region", { name: "Warnings" });
    expect(
      within(critical)
        .getAllByRole("article")
        .map((item) => item.getAttribute("aria-label")),
    ).toEqual(["Environment-credential login is enabled"]);
    expect(
      within(warnings)
        .getAllByRole("article")
        .map((item) => item.getAttribute("aria-label")),
    ).toEqual(["No Redis configured"]);
    expect(within(warnings).getByRole("link", { name: "Read the docs" })).toHaveAttribute(
      "href",
      "https://docs.litellm.ai/docs/proxy/redis_requirements",
    );
    expect(within(warnings).getByRole("link", { name: "Read the docs" })).toHaveAttribute("target", "_blank");
    expect(within(warnings).getByRole("link", { name: "Read the docs" })).toHaveAttribute(
      "rel",
      expect.stringContaining("noopener"),
    );
    expect(within(panel).queryByRole("article", { name: "Detailed debug logging is on" })).not.toBeInTheDocument();

    await user.click(within(panel).getByRole("button", { name: "Ignored and snoozed (2)" }));

    const debug = within(panel).getByRole("article", { name: "Detailed debug logging is on" });
    expect(within(debug).getByText(/^Snoozed until/)).toHaveTextContent(/^Snoozed until Oct 2[12], 2026/);
    expect(debug).toHaveTextContent(/by admin-2 on Sep (29|30), 2026/);
    const expired = within(panel).getByRole("article", {
      name: "Your LiteLLM Enterprise license expired on Oct 1, 2026",
    });
    expect(expired).toHaveTextContent("by admin-3");

    await user.hover(within(debug).getByText(/Snoozed until/));
    expect(await screen.findByText(/October 21, 2026 at 5:04:12\sPM UTC/)).toBeVisible();
  });

  it.each([
    { option: "Snooze for 1 day", body: { ignored: true, snooze_seconds: 86_400 }, badge: /Snoozed until/ },
    { option: "Snooze for 7 days", body: { ignored: true, snooze_seconds: 604_800 }, badge: /Snoozed until/ },
    { option: "Snooze for 30 days", body: { ignored: true, snooze_seconds: 2_592_000 }, badge: /Snoozed until/ },
    { option: "Ignore forever", body: { ignored: true }, badge: /^Ignored$/ },
  ])(
    "should let a proxy admin $option and move the warning to the ignored section",
    async ({ option, body, badge }) => {
      const user = userEvent.setup();
      proxy.warnings = [ENV_CREDENTIAL, NO_REDIS];
      renderHealth();
      const panel = await openPanel(user);

      await user.click(within(panel).getByRole("button", { name: "Actions for No Redis configured" }));
      await user.click(await screen.findByRole("menuitem", { name: option }));

      await waitFor(() => expect(proxy.patches).toEqual([{ id: "no_redis", body }]));
      await waitFor(() => expect(within(panel).queryByRole("region", { name: "Warnings" })).not.toBeInTheDocument());
      await user.click(within(panel).getByRole("button", { name: "Ignored and snoozed (1)" }));
      const item = within(panel).getByRole("article", { name: "No Redis configured" });
      expect(within(item).getByText(badge)).toBeVisible();
      expect(item).toHaveTextContent("by admin-1");

      await user.keyboard("{Escape}");
      expect(await screen.findByRole("button", { name: "Deployment health, 1 active warning" })).toBeVisible();
    },
  );

  it("should restore an ignored warning back into its severity group", async () => {
    const user = userEvent.setup();
    proxy.warnings = [{ ...ENV_CREDENTIAL, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") }];
    renderHealth();
    const panel = await openPanel(user);
    expect(within(panel).getByText("No active deployment warnings")).toBeVisible();

    await user.click(within(panel).getByRole("button", { name: "Ignored and snoozed (1)" }));
    await user.click(
      within(panel).getByRole("button", { name: "Actions for Environment-credential login is enabled" }),
    );
    await user.click(await screen.findByRole("menuitem", { name: "Restore" }));

    await waitFor(() => expect(proxy.patches).toEqual([{ id: "env_credential_login", body: { ignored: false } }]));
    const critical = await within(panel).findByRole("region", { name: "Critical" });
    expect(within(critical).getByRole("article", { name: "Environment-credential login is enabled" })).toBeVisible();

    await user.keyboard("{Escape}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Environment-credential login is enabled");
  });

  it("should show an env var ignore without a restore action", async () => {
    const user = userEvent.setup();
    proxy.warnings = [{ ...NO_REDIS, ignore: { source: "env_var", env_var: "LITELLM_DISABLE_NO_REDIS_WARNING" } }];
    renderHealth();
    const panel = await openPanel(user);

    await user.click(within(panel).getByRole("button", { name: "Ignored and snoozed (1)" }));

    const item = within(panel).getByRole("article", { name: "No Redis configured" });
    expect(item).toHaveTextContent("by the LITELLM_DISABLE_NO_REDIS_WARNING environment variable");
    expect(within(item).queryByRole("button")).not.toBeInTheDocument();
  });

  it("should replace the ignore menu with a database hint when no database is connected", async () => {
    const user = userEvent.setup();
    proxy.warnings = [NO_REDIS];
    proxy.ignoreAvailable = false;
    renderHealth();
    const panel = await openPanel(user);

    const item = within(panel).getByRole("article", { name: "No Redis configured" });
    expect(item).toHaveTextContent("Connect a database to ignore warnings");
    expect(within(item).queryByRole("button")).not.toBeInTheDocument();
  });

  it("should let a proxy admin viewer read warnings without ignore or restore actions", async () => {
    const user = userEvent.setup();
    proxy.warnings = [NO_REDIS, { ...ENV_CREDENTIAL, ignore: snoozedBy("admin-2", "2026-10-21T17:04:12Z") }];
    renderHealth("proxy_admin_viewer");
    const panel = await openPanel(user);
    await user.click(within(panel).getByRole("button", { name: "Ignored and snoozed (1)" }));

    const active = within(panel).getByRole("article", { name: "No Redis configured" });
    const ignored = within(panel).getByRole("article", { name: "Environment-credential login is enabled" });
    expect(ignored).toHaveTextContent("by admin-2");
    expect(within(active).queryByRole("button")).not.toBeInTheDocument();
    expect(within(ignored).queryByRole("button")).not.toBeInTheDocument();
    expect(active).not.toHaveTextContent("Connect a database");
  });

  it("should refetch when opened even while the icon is served from cache", async () => {
    const user = userEvent.setup();
    proxy.warnings = [NO_REDIS];
    renderHealth();
    await screen.findByRole("button", { name: "Deployment health, 1 active warning" });
    proxy.warnings = [NO_REDIS, DEBUG];

    const panel = await openPanel(user);

    expect(await within(panel).findByRole("article", { name: "Detailed debug logging is on" })).toBeVisible();
    expect(proxy.gets).toBe(2);
  });
});
