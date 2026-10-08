import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PluginModeProvider } from "@/contexts/PluginModeContext";
import { DashboardHeader } from "./DashboardHeader";

vi.mock("next/navigation", () => ({ usePathname: () => "/ui/logs" }));

beforeEach(() => {
  localStorage.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
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

async function openTools() {
  const user = userEvent.setup();
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <PluginModeProvider accessToken="test-session">
        <DashboardHeader />
      </PluginModeProvider>
    </QueryClientProvider>,
  );
  await user.click(screen.getByRole("button", { name: "More options" }));
  const tools = await screen.findByRole("dialog", { name: "Gateway tools" });
  return { user, tools };
}

describe("mobile header nested controls", () => {
  it.each(["mouse", "touch"] as const)(
    "selects a portalled gateway item with %s without dismissing More",
    async (pointer) => {
      const { user, tools } = await openTools();
      await user.click(within(tools).getByRole("button", { name: "AI Gateway" }));
      const item = await screen.findByRole("menuitem", { name: "Metrics" });
      if (pointer === "touch") {
        await user.pointer([
          { keys: "[TouchA>]", target: item },
          { keys: "[/TouchA]", target: item },
        ]);
      } else {
        await user.click(item);
      }
      expect(within(tools).getByRole("button", { name: "Metrics" })).toBeVisible();
      expect(screen.getByRole("dialog", { name: "Gateway tools" })).toBeVisible();
    },
  );

  it("lets a nested blog link receive the click before closing its menu", async () => {
    const { user, tools } = await openTools();
    await user.click(within(tools).getByRole("button", { name: "Blog" }));
    const link = await screen.findByRole("link", { name: /Gateway update/ });
    expect(link).toHaveAttribute("href", "https://example.com/update");
    await user.click(link);
    expect(screen.queryByRole("menu", { name: "Blog" })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Gateway tools" })).toBeVisible();
  });

  it("allows filtering and selecting a worker from the nested combobox", async () => {
    localStorage.setItem("litellm_selected_worker_id", "primary");
    const { user, tools } = await openTools();
    const worker = await within(tools).findByRole("combobox", { name: "Worker" });
    await user.clear(worker);
    await user.type(worker, "Backup");
    await user.click(await screen.findByRole("option", { name: "Backup worker" }));
    expect(localStorage.getItem("litellm_selected_worker_id")).toBeNull();
    expect(screen.getByRole("dialog", { name: "Gateway tools" })).toBeVisible();
  });
});
