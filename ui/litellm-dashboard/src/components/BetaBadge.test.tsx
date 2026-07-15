import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import BetaBadge from "./BetaBadge";

// Mock the hook directly
vi.mock("@/app/(dashboard)/hooks/useDisableShowBadges", () => ({
  useDisableShowBadges: vi.fn(),
}));

import { useDisableShowBadges } from "@/app/(dashboard)/hooks/useDisableShowBadges";

const mockUseDisableShowBadges = vi.mocked(useDisableShowBadges);

describe("BetaBadge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the badge when disableShowBadges is false", () => {
    mockUseDisableShowBadges.mockReturnValue(false);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render the badge when disableShowBadges is not set", () => {
    mockUseDisableShowBadges.mockReturnValue(false);

    render(<BetaBadge />);

    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("should render only children when disableShowBadges is true", () => {
    mockUseDisableShowBadges.mockReturnValue(true);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.queryByText("Beta")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render nothing when disableShowBadges is true and no children", () => {
    mockUseDisableShowBadges.mockReturnValue(true);

    const { container } = render(<BetaBadge />);

    expect(container.firstChild).toBeNull();
  });

  it("should render badge with dot instead of text when dot prop is true", () => {
    mockUseDisableShowBadges.mockReturnValue(false);

    render(<BetaBadge dot={true}>Test Content</BetaBadge>);

    expect(screen.queryByText("Beta")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render badge with 'Beta' text when dot prop is not provided (defaults to false)", () => {
    mockUseDisableShowBadges.mockReturnValue(false);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });
});
