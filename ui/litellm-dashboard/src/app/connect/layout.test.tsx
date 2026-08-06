import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ConnectLayout from "./layout";

const { mockUseAuthorized, state } = vi.hoisted(() => {
  const state = {
    accessToken: "token-123" as string | null,
    isAuthorized: true,
    isLoading: false,
  };
  return {
    state,
    mockUseAuthorized: vi.fn(() => ({
      accessToken: state.accessToken,
      isAuthorized: state.isAuthorized,
      isLoading: state.isLoading,
    })),
  };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: mockUseAuthorized }));
vi.mock("@/components/navbar", () => ({ default: () => <div data-testid="navbar" /> }));
vi.mock("@/contexts/ThemeContext", () => ({
  ThemeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

describe("ConnectLayout", () => {
  afterEach(() => {
    state.accessToken = "token-123";
    state.isAuthorized = true;
    state.isLoading = false;
  });

  it("renders the connect surface for an authorized user without any chat-ui flag", () => {
    render(
      <ConnectLayout>
        <div data-testid="page-content" />
      </ConnectLayout>,
    );
    expect(screen.getByTestId("navbar")).toBeInTheDocument();
    expect(screen.getByTestId("page-content")).toBeInTheDocument();
  });

  it("renders nothing when the user is not authorized", () => {
    state.isAuthorized = false;
    render(
      <ConnectLayout>
        <div data-testid="page-content" />
      </ConnectLayout>,
    );
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("navbar")).not.toBeInTheDocument();
  });

  it("renders nothing while authorization is still loading", () => {
    state.isLoading = true;
    render(
      <ConnectLayout>
        <div data-testid="page-content" />
      </ConnectLayout>,
    );
    expect(screen.queryByTestId("page-content")).not.toBeInTheDocument();
    expect(screen.queryByTestId("navbar")).not.toBeInTheDocument();
  });
});
