import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import NewBadge from "./NewBadge";

// Mock the hook directly
vi.mock("@/app/(dashboard)/hooks/useDisableShowNewBadge", () => ({
  useDisableShowNewBadge: vi.fn(),
}));

import { useDisableShowNewBadge } from "@/app/(dashboard)/hooks/useDisableShowNewBadge";

const mockUseDisableShowNewBadge = vi.mocked(useDisableShowNewBadge);

describe("NewBadge", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the badge when disableShowNewBadge is false", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge>Test Content</NewBadge>);

    expect(screen.getByText("New")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render the badge when disableShowNewBadge is not set", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge />);

    expect(screen.getByText("New")).toBeInTheDocument();
  });

  it("should render only children when disableShowNewBadge is true", () => {
    mockUseDisableShowNewBadge.mockReturnValue(true);

    render(<NewBadge>Test Content</NewBadge>);

    expect(screen.queryByText("New")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render nothing when disableShowNewBadge is true and no children", () => {
    mockUseDisableShowNewBadge.mockReturnValue(true);

    const { container } = render(<NewBadge />);

    expect(container.firstChild).toBeNull();
  });

  it("should render badge with dot when dot prop is true", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge dot={true}>Test Content</NewBadge>);

    expect(screen.queryByText("New")).not.toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render badge with 'New' text when dot prop is false", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge dot={false}>Test Content</NewBadge>);

    expect(screen.getByText("New")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render badge with 'New' text when dot prop is not provided (defaults to false)", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge>Test Content</NewBadge>);

    expect(screen.getByText("New")).toBeInTheDocument();
    expect(screen.getByText("Test Content")).toBeInTheDocument();
  });

  it("should render badge with dot when dot is true and no children", () => {
    mockUseDisableShowNewBadge.mockReturnValue(false);

    render(<NewBadge dot={true} />);

    expect(screen.queryByText("New")).not.toBeInTheDocument();
  });
});
