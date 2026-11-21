import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import { Sidebar, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { GatewayLensNavigation } from "./GatewayLensNavigation";

afterEach(() => vi.unstubAllGlobals());

it("switches Lens pages while preserving workspace, agent, and date filters", async () => {
  const onUrlUpdate = vi.fn();
  const scope = {
    organization_id: "org-a",
    team_id: "team-a",
    project_id: "project-a",
    api_key_hash: "key-hash",
    agent: "codex",
    hours: "24",
    from: "100",
    to: "200",
  };
  const initialParams = { ...scope, tab: "traces", trace: "old-trace", trace_ref: "old-reference" };
  renderWithProviders(
    <SidebarProvider>
      <GatewayLensNavigation />
    </SidebarProvider>,
    {
      searchParams: new URLSearchParams(initialParams),
      onUrlUpdate,
    },
  );
  expect(screen.getByRole("button", { name: "Traces" })).toHaveAttribute("aria-current", "page");
  fireEvent.click(screen.getByRole("button", { name: "Evals" }));
  await waitFor(() => expect(onUrlUpdate).toHaveBeenCalledOnce());
  const query = new URLSearchParams(onUrlUpdate.mock.lastCall?.[0].queryString);
  expect(Object.fromEntries(query)).toEqual({ ...scope, tab: "evals" });
  expect(screen.getByRole("button", { name: "Evals" })).toHaveAttribute("aria-current", "page");
});

it("keeps collapsed Lens navigation accessible by name", () => {
  renderWithProviders(
    <SidebarProvider defaultOpen={false}>
      <GatewayLensNavigation />
    </SidebarProvider>,
  );
  expect(screen.getByRole("button", { name: "Traces" })).toHaveAttribute("title", "Traces");
  expect(screen.getByRole("button", { name: "Settings" })).toHaveAttribute("title", "Settings");
});

it("closes the mobile sidebar after choosing a Lens page", async () => {
  vi.stubGlobal("innerWidth", 390);
  renderWithProviders(
    <SidebarProvider>
      <SidebarTrigger />
      <Sidebar>
        <GatewayLensNavigation />
      </Sidebar>
    </SidebarProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: "Toggle Sidebar" }));
  const drawer = await screen.findByRole("dialog", { name: "Sidebar" });
  fireEvent.click(within(drawer).getByRole("button", { name: "Traces" }));
  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Sidebar" })).not.toBeInTheDocument());
});
