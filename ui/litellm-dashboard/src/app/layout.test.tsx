import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import RootLayout from "./layout";

vi.mock("./globals.css", () => ({}));

vi.mock("next/font/google", () => ({
  Inter: () => ({ className: "inter" }),
}));

vi.mock("nuqs/adapters/next/app", () => ({
  NuqsAdapter: ({ children }: { children: React.ReactNode }) => <div data-testid="nuqs">{children}</div>,
}));

vi.mock("@/contexts/ReactQueryProvider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <div data-testid="query">{children}</div>,
}));

vi.mock("@/i18n/I18nProvider", () => ({
  I18nProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="i18n">{children}</div>,
}));

vi.mock("@/contexts/AntdGlobalProvider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <div data-testid="antd">{children}</div>,
}));

vi.mock("@/contexts/AuthContext", () => ({
  AuthProvider: ({ children }: { children: React.ReactNode }) => <div data-testid="auth">{children}</div>,
}));

describe("RootLayout", () => {
  it("provides localization before Ant Design and every route child", () => {
    const layout = RootLayout({ children: <div data-testid="route" /> });
    const body = layout.props.children;

    render(body.props.children);

    expect(screen.getByTestId("i18n")).toContainElement(screen.getByTestId("antd"));
    expect(screen.getByTestId("antd")).toContainElement(screen.getByTestId("route"));
  });
});
