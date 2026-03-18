import { renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { DebugWarningBanner } from "./DebugWarningBanner";

const mockUseHealthReadiness = vi.fn();
vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness", () => ({
  useHealthReadiness: () => mockUseHealthReadiness(),
}));

describe("DebugWarningBanner", () => {
  afterEach(() => {
    vi.resetAllMocks();
  });

  it("should render", () => {
    mockUseHealthReadiness.mockReturnValue({ data: { is_detailed_debug: true } });
    renderWithProviders(<DebugWarningBanner />);
    expect(screen.getByText(/Performance Warning/i)).toBeInTheDocument();
  });

  it("should render nothing when debug mode is disabled", () => {
    mockUseHealthReadiness.mockReturnValue({ data: { is_detailed_debug: false } });
    const { container } = renderWithProviders(<DebugWarningBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("should render nothing when health data is undefined", () => {
    mockUseHealthReadiness.mockReturnValue({ data: undefined });
    const { container } = renderWithProviders(<DebugWarningBanner />);
    expect(container.firstChild).toBeNull();
  });

  it("should mention LITELLM_LOG=DEBUG in the description", () => {
    mockUseHealthReadiness.mockReturnValue({ data: { is_detailed_debug: true } });
    renderWithProviders(<DebugWarningBanner />);
    expect(screen.getByText(/LITELLM_LOG=DEBUG/)).toBeInTheDocument();
  });
});
