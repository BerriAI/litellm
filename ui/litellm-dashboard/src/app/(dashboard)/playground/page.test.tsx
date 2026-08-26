import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
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

describe("PlaygroundPage role guard", () => {
  beforeEach(() => {
    authState.userRole = "Admin";
  });

  it.each(["Internal Viewer", "Admin Viewer"])("blocks the entire playground for %s", (role) => {
    authState.userRole = role;
    render(<PlaygroundPage />);

    expect(screen.getByText("Access Denied")).toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByTestId("chat-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("compare-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("compliance-ui")).not.toBeInTheDocument();
    expect(screen.queryByTestId("agent-builder")).not.toBeInTheDocument();
  });

  it.each(["Admin", "Internal User", "Org Admin"])("renders the playground for %s", (role) => {
    authState.userRole = role;
    render(<PlaygroundPage />);

    expect(screen.queryByText("Access Denied")).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Chat" })).toBeInTheDocument();
    expect(screen.getByTestId("chat-ui")).toBeInTheDocument();
  });
});
