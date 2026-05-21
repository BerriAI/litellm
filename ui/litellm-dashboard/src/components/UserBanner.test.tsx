import { renderWithProviders, screen } from "../../tests/test-utils";
import { vi } from "vitest";
import { UserBanner } from "./UserBanner";

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: vi.fn(),
}));

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";

describe("UserBanner", () => {
  it("should render published banner message", () => {
    vi.mocked(useUISettings).mockReturnValue({
      data: {
        values: {
          user_banner_enabled: true,
          user_banner_message: "Scheduled maintenance tonight at 10 PM UTC.",
          user_banner_type: "warning",
        },
      },
    } as any);

    renderWithProviders(<UserBanner />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(
      screen.getByText("Scheduled maintenance tonight at 10 PM UTC."),
    ).toBeInTheDocument();
  });

  it("should render nothing when banner is unpublished", () => {
    vi.mocked(useUISettings).mockReturnValue({
      data: {
        values: {
          user_banner_enabled: false,
          user_banner_message: "Hidden message",
          user_banner_type: "info",
        },
      },
    } as any);

    const { container } = renderWithProviders(<UserBanner />);

    expect(container).toBeEmptyDOMElement();
  });

  it("should render nothing when message is blank", () => {
    vi.mocked(useUISettings).mockReturnValue({
      data: {
        values: {
          user_banner_enabled: true,
          user_banner_message: "   ",
          user_banner_type: "info",
        },
      },
    } as any);

    const { container } = renderWithProviders(<UserBanner />);

    expect(container).toBeEmptyDOMElement();
  });
});
