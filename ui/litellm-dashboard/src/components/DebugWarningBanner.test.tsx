import { renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { DebugWarningBanner } from "./DebugWarningBanner";

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails", () => ({
  useHealthReadinessDetails: vi.fn(),
}));

import { useHealthReadinessDetails } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadinessDetails";

describe("DebugWarningBanner", () => {
  it("should render", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner accessToken="token" userRole="Admin" />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("should show warning when detailed debug mode is active", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner accessToken="token" userRole="Admin" />);
    expect(screen.getByText(/Performance Warning: Detailed Debug Mode Active/i)).toBeInTheDocument();
  });

  it("should mention LITELLM_LOG=DEBUG in the description", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner accessToken="token" userRole="Admin" />);
    expect(screen.getByText("LITELLM_LOG=DEBUG")).toBeInTheDocument();
  });

  it("should render nothing when is_detailed_debug is false", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: { is_detailed_debug: false } } as any);
    const { container } = renderWithProviders(<DebugWarningBanner accessToken="token" userRole="Admin" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when health data is undefined", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: undefined } as any);
    const { container } = renderWithProviders(<DebugWarningBanner accessToken="token" userRole="Admin" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should pass accessToken and userRole to the readiness hook", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: undefined } as any);
    renderWithProviders(<DebugWarningBanner accessToken="my-token" userRole="Admin" />);
    expect(useHealthReadinessDetails).toHaveBeenCalledWith("my-token", "Admin");
  });

  it("should pass a null accessToken through (disables the hook)", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: undefined } as any);
    renderWithProviders(<DebugWarningBanner accessToken={null} userRole="Admin" />);
    expect(useHealthReadinessDetails).toHaveBeenCalledWith(null, "Admin");
  });

  it("should pass a non-admin role through so the hook can gate the admin-only call", () => {
    vi.mocked(useHealthReadinessDetails).mockReturnValue({ data: undefined } as any);
    renderWithProviders(<DebugWarningBanner accessToken="my-token" userRole="Internal User" />);
    expect(useHealthReadinessDetails).toHaveBeenCalledWith("my-token", "Internal User");
  });
});
