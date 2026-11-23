import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import type { ComponentProps } from "react";
import { expect, it, vi } from "vitest";
import Sidebar from "@/components/LeftNav";
import AppSidebar from "./AppSidebar";

const { getSettings } = vi.hoisted(() => ({
  getSettings: vi.fn().mockResolvedValue({ values: { allow_agents_for_team_admins: true } }),
}));

vi.mock("@/components/networking", () => ({ getUiSettings: getSettings, getProxyBaseUrl: () => "http://localhost" }));
vi.mock("@/components/LeftNav", () => ({
  default: ({ allowAgentsForTeamAdmins }: ComponentProps<typeof Sidebar>) => (
    <nav>{allowAgentsForTeamAdmins && <a href="/agents">Agents</a>}</nav>
  ),
}));

it("reuses loaded navigation settings immediately when the mobile sidebar mounts again", async () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const shell = (mobileOpen: boolean) => (
    <QueryClientProvider client={client}>
      <section aria-label="Desktop navigation">
        <AppSidebar />
      </section>
      {mobileOpen && (
        <section aria-label="Mobile navigation">
          <AppSidebar />
        </section>
      )}
    </QueryClientProvider>
  );
  const { rerender } = render(shell(false));
  expect(await screen.findByRole("link", { name: "Agents" })).toBeVisible();

  for (const open of [true, false, true]) {
    rerender(shell(open));
    if (open) {
      expect(
        within(screen.getByRole("region", { name: "Mobile navigation" })).getByRole("link", { name: "Agents" }),
      ).toBeVisible();
    }
  }
  expect(getSettings).toHaveBeenCalledTimes(1);
});
