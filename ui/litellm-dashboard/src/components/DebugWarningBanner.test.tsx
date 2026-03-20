import { renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { DebugWarningBanner } from "./DebugWarningBanner";

vi.mock("@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness", () => ({
  useHealthReadiness: vi.fn(),
}));

import { useHealthReadiness } from "@/app/(dashboard)/hooks/healthReadiness/useHealthReadiness";

describe("DebugWarningBanner", () => {
  it("should render", () => {
    vi.mocked(useHealthReadiness).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("should show warning when detailed debug mode is active", () => {
    vi.mocked(useHealthReadiness).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner />);
    expect(screen.getByText(/Performance Warning: Detailed Debug Mode Active/i)).toBeInTheDocument();
  });

  it("should mention LITELLM_LOG=DEBUG in the description", () => {
    vi.mocked(useHealthReadiness).mockReturnValue({ data: { is_detailed_debug: true } } as any);
    renderWithProviders(<DebugWarningBanner />);
    expect(screen.getByText("LITELLM_LOG=DEBUG")).toBeInTheDocument();
  });

  it("should render nothing when is_detailed_debug is false", () => {
    vi.mocked(useHealthReadiness).mockReturnValue({ data: { is_detailed_debug: false } } as any);
    const { container } = renderWithProviders(<DebugWarningBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when health data is undefined", () => {
    vi.mocked(useHealthReadiness).mockReturnValue({ data: undefined } as any);
    const { container } = renderWithProviders(<DebugWarningBanner />);
    expect(container).toBeEmptyDOMElement();
  });
});
