import React from "react";
import { render, screen } from "@testing-library/react";
import { UserBanner } from "./UserBanner";

// Mock the useUISettings hook
jest.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: jest.fn(),
}));

import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
const mockUseUISettings = useUISettings as jest.MockedFunction<typeof useUISettings>;

describe("UserBanner", () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders nothing when banner is disabled", () => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: { user_banner_enabled: false, user_banner_message: "Hello", user_banner_type: "info" },
        field_schema: {},
      },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    const { container } = render(<UserBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when banner is enabled but message is empty", () => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: { user_banner_enabled: true, user_banner_message: "", user_banner_type: "warning" },
        field_schema: {},
      },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    const { container } = render(<UserBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the banner message when enabled with non-empty message", () => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: {
          user_banner_enabled: true,
          user_banner_message: "Scheduled maintenance on Friday",
          user_banner_type: "warning",
        },
        field_schema: {},
      },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    render(<UserBanner />);
    expect(screen.getByText("Scheduled maintenance on Friday")).toBeInTheDocument();
  });

  it("falls back to 'info' type for unrecognised banner types", () => {
    mockUseUISettings.mockReturnValue({
      data: {
        values: {
          user_banner_enabled: true,
          user_banner_message: "Hello",
          user_banner_type: "critical", // not a valid type
        },
        field_schema: {},
      },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    // Should not throw; falls back to "info"
    render(<UserBanner />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });
});
