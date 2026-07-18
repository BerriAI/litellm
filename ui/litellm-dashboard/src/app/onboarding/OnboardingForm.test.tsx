import { act, fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OnboardingForm } from "./OnboardingForm";

const mockUseOnboardingCredentials = vi.fn();
const mockClaimToken = vi.fn();

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams("invitation_id=inv-123"),
}));

vi.mock("jwt-decode", () => ({
  jwtDecode: vi.fn(() => ({
    user_email: "alice@example.com",
    user_id: "user-1",
    key: "access-tok",
  })),
}));

vi.mock("@/app/(dashboard)/hooks/onboarding/useOnboarding", () => ({
  useOnboardingCredentials: (...args: unknown[]) => mockUseOnboardingCredentials(...args),
  useClaimOnboardingToken: () => ({ mutate: mockClaimToken, isPending: false }),
}));

vi.mock("@/components/networking", () => ({
  getProxyBaseUrl: vi.fn(() => ""),
}));

vi.mock("./OnboardingLoadingView", () => ({
  OnboardingLoadingView: () => <div data-testid="loading-view">Loading</div>,
}));

vi.mock("./OnboardingErrorView", () => ({
  OnboardingErrorView: () => <div data-testid="error-view">Error</div>,
}));

vi.mock("./OnboardingFormBody", () => ({
  OnboardingFormBody: ({
    variant,
    userEmail,
    claimError,
    onSubmit,
  }: {
    variant: string;
    userEmail: string;
    claimError: string | null;
    onSubmit: (formValues: { password: string }) => void;
  }) => (
    <div data-testid="form-body" data-variant={variant} data-email={userEmail}>
      Form Body
      <button type="button" onClick={() => onSubmit({ password: "NewP@ssw0rd" })}>
        Submit
      </button>
      {claimError ? <div data-testid="claim-error">{claimError}</div> : null}
    </div>
  ),
}));

describe("OnboardingForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/";
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/ui";
    sessionStorage.clear();
  });

  it("should render loading view when credentials are loading", () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
    });

    render(<OnboardingForm variant="signup" />);

    expect(screen.getByTestId("loading-view")).toBeInTheDocument();
  });

  it("should render error view when credentials fail to load", () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });

    render(<OnboardingForm variant="signup" />);

    expect(screen.getByTestId("error-view")).toBeInTheDocument();
  });

  it("should render form body with decoded email when credentials are loaded", () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: { token: "fake-jwt-token" },
      isLoading: false,
      isError: false,
    });

    render(<OnboardingForm variant="signup" />);

    expect(screen.getByTestId("form-body")).toBeInTheDocument();
    expect(screen.getByTestId("form-body")).toHaveAttribute("data-email", "alice@example.com");
  });

  it("should pass variant prop to OnboardingFormBody", () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: { token: "fake-jwt-token" },
      isLoading: false,
      isError: false,
    });

    render(<OnboardingForm variant="reset_password" />);

    expect(screen.getByTestId("form-body")).toHaveAttribute("data-variant", "reset_password");
  });

  it("should overwrite a prior admin sessionStorage token after successful claim", async () => {
    // Simulate the prior admin session that the inviter left behind in the same tab.
    sessionStorage.setItem("token", "ADMIN_SESSION_TOKEN");

    mockUseOnboardingCredentials.mockReturnValue({
      data: { token: "fake-jwt-token" },
      isLoading: false,
      isError: false,
    });
    mockClaimToken.mockImplementation((_params, options) => {
      options.onSuccess({ token: "NEW_USER_TOKEN" });
    });

    render(<OnboardingForm variant="signup" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    });

    // After signup the new user's token must replace the admin's sessionStorage entry,
    // otherwise the HttpOnly-fallback path in getCookie() keeps returning the admin token.
    expect(sessionStorage.getItem("token")).toBe("NEW_USER_TOKEN");
  });

  it("should set the new token cookie at path=/ui after successful claim", async () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: { token: "fake-jwt-token" },
      isLoading: false,
      isError: false,
    });
    mockClaimToken.mockImplementation((_params, options) => {
      options.onSuccess({ token: "NEW_USER_TOKEN" });
    });

    const cookieSpy = vi.spyOn(document, "cookie", "set");
    render(<OnboardingForm variant="signup" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    });

    // The /ui path is what storeLoginToken writes to (and what LoginPage reads from);
    // a cookie at path=/ alone leaves any pre-existing /ui-scoped admin cookie winning.
    const newTokenCookieAtUiPath = cookieSpy.mock.calls.some(([value]) => {
      const v = String(value);
      return v.includes("token=NEW_USER_TOKEN") && v.includes("path=/ui");
    });
    expect(newTokenCookieAtUiPath).toBe(true);

    cookieSpy.mockRestore();
  });

  it("should show claim error when claim response is missing final token", async () => {
    mockUseOnboardingCredentials.mockReturnValue({
      data: { token: "fake-jwt-token" },
      isLoading: false,
      isError: false,
    });
    mockClaimToken.mockImplementation((_params, options) => {
      options.onSuccess({});
    });

    render(<OnboardingForm variant="signup" />);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: "Submit" }));
    });

    expect(screen.getByTestId("claim-error")).toHaveTextContent("Failed to start session");
    expect(document.cookie).not.toContain("fake-jwt-token");
  });
});
