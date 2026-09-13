import { fireEvent, renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { EnvCredentialLoginWarningBanner } from "./EnvCredentialLoginWarningBanner";
import type { HealthReadinessDetailsResponse } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import type { UseQueryResult } from "@tanstack/react-query";

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));
vi.mock("@/contexts/AuthContext", () => ({
  useAuth: vi.fn(),
}));

import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import { useAuth } from "@/contexts/AuthContext";

const mockDetails = (data: Partial<HealthReadinessDetailsResponse> | undefined) => {
  vi.mocked(useHealthReadinessDetails).mockReturnValue({ data } as UseQueryResult<HealthReadinessDetailsResponse>);
};

const mockRole = (userRole: string) => {
  vi.mocked(useAuth).mockReturnValue({ userRole } as ReturnType<typeof useAuth>);
};

describe("EnvCredentialLoginWarningBanner", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("should hide the banner when dismissed and stay hidden on remount", () => {
    mockRole("Admin");
    mockDetails({ status: "healthy", show_env_credential_login_warning: true });
    const first = renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    fireEvent.click(screen.getByRole("button", { name: "Dismiss banner" }));
    expect(first.container).toBeEmptyDOMElement();

    first.unmount();
    const second = renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(second.container).toBeEmptyDOMElement();
  });

  it("should warn an admin when env-credential login is enabled", () => {
    mockRole("Admin");
    mockDetails({ status: "healthy", show_env_credential_login_warning: true });
    renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Environment-credential login is enabled")).toBeInTheDocument();
  });

  it("should tell the admin to create a regular admin account before disabling", () => {
    mockRole("Admin");
    mockDetails({ status: "healthy", show_env_credential_login_warning: true });
    renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(screen.getByText(/First create a regular admin account/i)).toBeInTheDocument();
    expect(screen.getByText("general_settings.disable_env_credential_login: true")).toBeInTheDocument();
  });

  it("should warn an admin viewer too", () => {
    mockRole("Admin Viewer");
    mockDetails({ status: "healthy", show_env_credential_login_warning: true });
    renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("should render nothing for a non-admin even when the proxy reports the warning", () => {
    mockRole("Internal User");
    mockDetails({ status: "healthy", show_env_credential_login_warning: true });
    const { container } = renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when env-credential login is disabled", () => {
    mockRole("Admin");
    mockDetails({ status: "healthy", show_env_credential_login_warning: false });
    const { container } = renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when readiness details are unavailable", () => {
    mockRole("Admin");
    mockDetails(undefined);
    const { container } = renderWithProviders(<EnvCredentialLoginWarningBanner accessToken={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should pass the access token to the readiness hook", () => {
    mockRole("Admin");
    mockDetails(undefined);
    renderWithProviders(<EnvCredentialLoginWarningBanner accessToken="my-token" />);
    expect(useHealthReadinessDetails).toHaveBeenCalledWith("my-token");
  });
});
