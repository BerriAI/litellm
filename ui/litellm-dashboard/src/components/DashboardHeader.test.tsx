import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { DashboardHeader } from "./DashboardHeader";
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
vi.mock("@/components/liteadmin/LiteAdmin", () => ({ default: () => <button>Aria Agent</button> }));

const renderDashboardHeader = () =>
  render(
    <CommandPaletteProvider>
      <DashboardHeader accessToken={null} />
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

  it("titles the dashboard root as Home", () => {
    state.pathname = "/ui/";
    renderDashboardHeader();

    expect(screen.getByText("Home")).toBeInTheDocument();
  });

  it("names the Aria workspace in the header", () => {
    state.pathname = "/ui/aria";
    renderDashboardHeader();
    expect(screen.getByRole("heading", { name: "Aria Agent" })).toBeInTheDocument();
  });

  it("keeps the scoped key search available", () => {
    state.pathname = "/ui/api-keys";
    renderDashboardHeader();
    expect(screen.getByRole("button", { name: "Search keys…" })).toBeInTheDocument();
  });
});
