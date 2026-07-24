import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import BetaBadge from "./BetaBadge";

// Mock the hook directly
vi.mock("@/app/(dashboard)/hooks/useDisableShowNewBadge", () => ({
  useDisableShowNewBadge: vi.fn(),
}));

import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";

const mockUseDisableShowNewBadge = vi.mocked(useDisableShowNewBadge);

describe("BetaBadge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the badge when disableShowNewBadge is false", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render the badge when disableShowNewBadge is not set", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<BetaBadge />);

    expect(screen.getByText("Beta")).toBeInTheDocument();
  });

  it("should render only children when disableShowNewBadge is true", () => {
    mockUseDisableShowNewBadge.mockReturnValue(true);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.queryByText("Beta")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render nothing when disableShowNewBadge is true and no children", () => {
    mockUseDisableShowNewBadge.mockReturnValue(true);

    const { container } = render(<BetaBadge />);

    expect(container.firstChild).toBeNull();
  });

  it("should render badge with dot instead of text when dot prop is true", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<BetaBadge dot={true}>Test Content</BetaBadge>);

    expect(screen.queryByText("Beta")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render badge with 'Beta' text when dot prop is not provided (defaults to false)", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<BetaBadge>Test Content</BetaBadge>);

    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });
});
