import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import PlaygroundPage from "./page";

const authState = { userRole: "Admin" };

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({
    token: "token-1",
    accessToken: "sk-test",
    userId: "user-1",
    userRole: authState.userRole,
    isViewOnly: ["Admin Viewer", "Internal Viewer"].includes(authState.userRole),
    disabledPersonalKeyCreation: false,
  }),
}));

vi.mock("@/utils/proxyUtils", () => ({
  fetchProxySettings: vi.fn().mockResolvedValue(null),
}));

vi.mock("@/app/(dashboard)/playground/components/chat_ui/ChatUI", () => ({
  default: () => <div data-testid="chat-ui" />,
}));

vi.mock("@/app/(dashboard)/playground/components/compareUI/CompareUI", () => ({
  default: () => <div data-testid="compare-ui" />,
}));

vi.mock("@/app/(dashboard)/playground/components/complianceUI/ComplianceUI", () => ({
  default: () => <div data-testid="compliance-ui" />,
}));

vi.mock("@/app/(dashboard)/playground/components/chat_ui/AgentBuilderView", () => ({
  default: () => <div data-testid="agent-builder" />,
}));

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

beforeEach(() => {
  authState.userRole = "Admin";
});

describe("PlaygroundPage role guard", () => {
  it.each(["Internal Viewer", "Admin Viewer"])("blocks the entire playground for %s", (role) => {
    authState.userRole = role;
    renderWithProviders(<PlaygroundPage />);

    expect(screen.getByText("Access Denied")).toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("chat-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("compare-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("compliance-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("agent-builder")).not.toBeInTheDocument();
  });

  it.each(["Admin", "Internal User", "Org Admin"])("renders the playground for %s", (role) => {
    authState.userRole = role;
    renderWithProviders(<PlaygroundPage />);

    expect(screen.queryByText("Access Denied")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chat" })).toBeInTheDocument();
    expect(screen.getByTestId("chat-ui")).toBeInTheDocument();
  });
});

describe("PlaygroundPage ?tab= deep link", () => {
  it("opens on Chat when the URL has no tab", () => {
    renderWithProviders(<PlaygroundPage />);

    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true");
  });

  it("activates the tab named in ?tab=", () => {
    renderWithProviders(<PlaygroundPage />, { searchParams: { tab: "compare" } });

    expect(screen.getByRole("tab", { name: "Compare" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "false");
  });

  it("falls back to Chat when ?tab= is not a playground tab", () => {
    renderWithProviders(<PlaygroundPage />, { searchParams: { tab: "settings" } });

    expect(screen.getByRole("tab", { name: "Chat" })).toHaveAttribute("aria-selected", "true");
  });

  it("clicking a tab writes ?tab= with history replace", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<PlaygroundPage />, { onUrlUpdate });

    await user.click(screen.getByRole("tab", { name: "Compliance" }));

    expect(await screen.findByRole("tab", { name: "Compliance", selected: true })).toBeInTheDocument();
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("compliance"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
  });
});
