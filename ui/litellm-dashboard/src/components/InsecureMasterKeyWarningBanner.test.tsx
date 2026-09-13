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
    expect(screen.getByText("The master key is the docs example key sk-1234")).toBeInTheDocument();
  });

  it("should warn when the proxy reports no master key", () => {
    mockDetails({ status: "healthy", insecure_master_key_reason: "missing" });
    renderWithProviders(<InsecureMasterKeyWarningBanner accessToken="token" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("No master key is set")).toBeInTheDocument();
    expect(screen.getByText(/accepted without authentication/)).toBeInTheDocument();
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
