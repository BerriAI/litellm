import { render, screen } from "@testing-library/react";
import { MessageSquare } from "lucide-react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NudgePrompt } from "./NudgePrompt";

vi.mock("@/app/(dashboard)/hooks/useDisableShowPrompts", () => ({
  useDisableShowPrompts: vi.fn(),
}));

vi.mock("@/utils/localStorageUtils", () => ({
  setLocalStorageItem: vi.fn(),
  emitLocalStorageChange: vi.fn(),
  LOCAL_STORAGE_EVENT: "local-storage-change",
}));

import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { emitLocalStorageChange, setLocalStorageItem } from "@/utils/localStorageUtils";

const mockUseDisableShowPrompts = vi.mocked(useDisableShowPrompts);
const mockSetLocalStorageItem = vi.mocked(setLocalStorageItem);
const mockEmitLocalStorageChange = vi.mocked(emitLocalStorageChange);

const defaultProps = {
  onOpen: vi.fn(),
  onDismiss: vi.fn(),
  isVisible: true,
  title: "Test Title",
  description: "Test Description",
  buttonText: "Open Modal",
  icon: MessageSquare,
  accentColor: "#3b82f6",
};

describe("NudgePrompt", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseDisableShowPrompts.mockReturnValue(false);
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("should render", () => {
    render(<NudgePrompt {...defaultProps} />);

    expect(screen.getByText("Test Title")).toBeInTheDocument();
  });

  it("should render with all provided props", () => {
    const { container } = render(<NudgePrompt {...defaultProps} />);

    expect(screen.getByText("Test Title")).toBeInTheDocument();
    expect(screen.getByText("Test Description")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open Modal" })).toBeInTheDocument();
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("should not render when isVisible is false", () => {
    render(<NudgePrompt {...defaultProps} isVisible={false} />);

    expect(screen.queryByText("Test Title")).not.toBeInTheDocument();
  });

  it("should not render when disableShowPrompts is true", () => {
    mockUseDisableShowPrompts.mockReturnValue(true);

    render(<NudgePrompt {...defaultProps} />);

    expect(screen.queryByText("Test Title")).not.toBeInTheDocument();
  });

  it("should display progress bar with correct accent color", () => {
    const { container } = render(<NudgePrompt {...defaultProps} accentColor="#ff0000" />);

    const progressBar = container.querySelector("div[style*='width']");
    expect(progressBar).toHaveStyle({ backgroundColor: "#ff0000" });
  });

  it("should reset progress when isVisible becomes false", () => {
    const { rerender, container } = render(<NudgePrompt {...defaultProps} />);

    vi.advanceTimersByTime(5000);

    rerender(<NudgePrompt {...defaultProps} isVisible={false} />);

    rerender(<NudgePrompt {...defaultProps} isVisible={true} />);

    const progressBar = container.querySelector("div[style*='width']");
    expect(progressBar?.getAttribute("style")).toContain("width: 100%");
  });

  it("should apply custom button style when provided", () => {
    const buttonStyle = { backgroundColor: "#custom-color" };
    render(<NudgePrompt {...defaultProps} buttonStyle={buttonStyle} />);

    const openButton = screen.getByRole("button", { name: "Open Modal" });
    expect(openButton).toHaveStyle(buttonStyle);
  });
});
