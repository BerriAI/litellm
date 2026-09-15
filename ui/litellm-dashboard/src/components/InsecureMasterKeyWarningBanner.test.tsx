import { renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { InsecureMasterKeyWarningBanner } from "./InsecureMasterKeyWarningBanner";
import type { HealthReadinessDetailsResponse } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";
import type { UseQueryResult } from "@tanstack/react-query";

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));

import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

const mockDetails = (data: Partial<HealthReadinessDetailsResponse> | undefined) => {
  vi.mocked(useHealthReadinessDetails).mockReturnValue({ data } as UseQueryResult<HealthReadinessDetailsResponse>);
};

describe("InsecureMasterKeyWarningBanner", () => {
  it("should warn when the proxy reports the docs example key", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: "example_key" });
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("The master key has not been set")).toBeInTheDocument();
    expect(screen.getByText(/docs example value sk-1234/)).toBeInTheDocument();
    expect(screen.getByText(/store or use upstream credentials or manage virtual keys/)).toBeInTheDocument();
  });

  it("should warn when the proxy reports no master key", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: "missing" });
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("The master key has not been set")).toBeInTheDocument();
    expect(screen.getByText(/store or use upstream credentials or manage virtual keys/)).toBeInTheDocument();
  });

  it("should note stored credentials are locked when the proxy reports them", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: "example_key", stored_credentials_locked: true });
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(screen.getByText(/Credentials already stored on this proxy/)).toBeInTheDocument();
  });

  it("should not mention stored credentials when none are locked", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: "missing", stored_credentials_locked: false });
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(screen.queryByText(/Credentials already stored on this proxy/)).not.toBeInTheDocument();
  });

  it("should render nothing when the configured key is strong", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: null });
    const { container } = renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when readiness details are unavailable", () => {
    mockDetails(undefined);
    const { container } = renderWithProviders(<InsecureMasterKeyWarningBanner accessToken={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should pass the access token to the readiness hook", () => {
    mockDetails(undefined);
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="my-token" />);
    expect(useHealthReadinessDetails).toHaveBeenCalledWith("my-token");
  });
});
