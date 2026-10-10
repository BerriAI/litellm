import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { DashboardHeader } from "./DashboardHeader";
import { NAV_PRODUCT_LINK_CLASS } from "@/components/Navbar/navProductLinkClass";
import { CommandPaletteProvider } from "@/components/CommandPalette/CommandPaletteProvider";

const { mockUsePluginMode, mockUseUISettings, state } = vi.hoisted(() => {
  const state = {
    plugins: [] as { name: string; display_name: string; url: string }[],
    enableChatUI: false,
    pathname: "/ui/logs",
    isDesktop: false,
  };
  return {
    state,
    mockUsePluginMode: vi.fn(() => ({ mode: "ai-gateway", setMode: vi.fn(), plugins: state.plugins })),
    mockUseUISettings: vi.fn(() => ({ data: { values: { enable_chat_ui: state.enableChatUI } } })),
  };
});

vi.mock("@/contexts/PluginModeContext", () => ({ usePluginMode: mockUsePluginMode }));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({ useUISettings: mockUseUISettings }));
vi.mock("next/navigation", () => ({ usePathname: () => state.pathname }));
vi.mock("usehooks-ts", () => ({ useMediaQuery: () => state.isDesktop }));
vi.mock("@/hooks/useWorker", () => ({ useWorker: () => ({ isControlPlane: false, selectedWorker: null }) }));
vi.mock("@/app/(dashboard)/hooks/useDisableShowPrompts", () => ({ useDisableShowPrompts: () => false }));
vi.mock("@/components/Navbar/BlogDropdown/BlogDropdown", () => ({ BlogDropdown: () => null }));
vi.mock("@/components/Navbar/CommunityEngagementButtons/CommunityEngagementButtons", () => ({
  CommunityEngagementButtons: () => null,
}));
vi.mock("@/components/Navbar/NotificationsBell/NotificationsBell", () => ({ NotificationsBell: () => null }));
vi.mock("@/components/Navbar/WorkerDropdown/WorkerDropdown", () => ({ default: () => null }));
vi.mock("@/components/liteadmin/LiteAdmin", () => ({ default: () => <button>LiteAdmin</button> }));

const renderDashboardHeader = () =>
  render(
    <CommandPaletteProvider>
      <DashboardHeader />
    </CommandPaletteProvider>,
  );

describe("DashboardHeader breadcrumb", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  afterEach(() => {
    state.plugins = [];
    state.enableChatUI = false;
    state.pathname = "/ui/logs";
    state.isDesktop = false;
    localStorage.clear();
  });

  it("titles the breadcrumb from the current route, not from a sidebar page id", () => {
    state.pathname = "/ui/models-and-endpoints";
    renderDashboardHeader();

    expect(screen.getByText("Models + Endpoints")).toBeInTheDocument();
  });

  it("titles the dashboard root as Discover", () => {
    state.pathname = "/ui/";
    renderDashboardHeader();

    expect(screen.getByText("Discover")).toBeInTheDocument();
  });

  it("roots the breadcrumb in the AI Gateway selector (with a Chat option) and drops the static section crumb when the selector is available", async () => {
    state.enableChatUI = true;
    renderDashboardHeader();

    expect(screen.getByText("Logs")).toBeInTheDocument();
    expect(screen.queryByText("Observability")).not.toBeInTheDocument();

    const selector = screen.getByRole("button", { name: /AI Gateway/i });
    act(() => {
      fireEvent.click(selector);
    });
    expect(await screen.findByText("Chat")).toBeInTheDocument();
  });

  it("keeps the AI Gateway selector at the root even when there is nothing to switch to (discovery)", () => {
    renderDashboardHeader();

    expect(screen.getByRole("button", { name: /AI Gateway/i })).toBeInTheDocument();
    expect(screen.getByText("Logs")).toBeInTheDocument();
    expect(screen.queryByText("Observability")).not.toBeInTheDocument();
  });

  it("styles Docs with the shared product-link class instead of a muted toolbar button", () => {
    renderDashboardHeader();

    const docs = screen.getByRole("link", { name: "Docs" });
    for (const cls of NAV_PRODUCT_LINK_CLASS.trim().split(/\s+/)) {
      expect(docs).toHaveClass(cls);
    }
    expect(docs).not.toHaveClass("text-muted-foreground");
  });

  it("renders the tools divider centered rather than stretched to the top of the row", () => {
    const { container } = renderDashboardHeader();

    const separators = container.querySelectorAll('[data-slot="separator"][data-orientation="vertical"]');
    expect(separators).toHaveLength(1);
    expect(separators[0].className).not.toMatch(/self-stretch/);
    expect(separators[0].className).toContain("data-vertical:self-center");
  });

  it("places LiteAdmin in the header tools ahead of Docs", () => {
    renderDashboardHeader();

    const liteAdmin = within(screen.getByRole("banner")).getByRole("button", { name: "LiteAdmin" });
    expect(liteAdmin.compareDocumentPosition(screen.getByRole("link", { name: "Docs" }))).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  });

  it("keeps the gateway selector and tools available from the compact header menu", async () => {
    renderDashboardHeader();
    fireEvent.click(screen.getByRole("button", { name: "More options" }));
    const tools = await screen.findByRole("dialog", { name: "Gateway tools" });
    expect(within(tools).getByRole("button", { name: "AI Gateway" })).toBeInTheDocument();
    expect(within(tools).getByRole("link", { name: "Docs" })).toBeInTheDocument();
    expect(within(tools).getByRole("button", { name: "LiteAdmin" })).toBeInTheDocument();
  });

  it("closes mobile tools when switching to desktop and keeps them closed when returning", async () => {
    const { rerender } = renderDashboardHeader();
    fireEvent.click(screen.getByRole("button", { name: "More options" }));
    expect(await screen.findByRole("dialog", { name: "Gateway tools" })).toBeInTheDocument();

    state.isDesktop = true;
    rerender(
      <CommandPaletteProvider>
        <DashboardHeader />
      </CommandPaletteProvider>,
    );
    expect(screen.queryByRole("dialog", { name: "Gateway tools" })).not.toBeInTheDocument();

    state.isDesktop = false;
    rerender(
      <CommandPaletteProvider>
        <DashboardHeader />
      </CommandPaletteProvider>,
    );
    expect(screen.queryByRole("dialog", { name: "Gateway tools" })).not.toBeInTheDocument();
  });
});
