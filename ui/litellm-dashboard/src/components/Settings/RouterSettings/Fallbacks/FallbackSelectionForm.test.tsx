import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FallbackSelectionForm } from "./FallbackSelectionForm";
import type { FallbackGroup } from "./FallbackGroupConfig";

const mockOnGroupsChange = vi.fn();
const AVAILABLE_MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"];

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    message: {
      ...actual.message,
      warning: vi.fn(),
    },
  };
});

describe("FallbackSelectionForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(Date, "now").mockReturnValue(1234567890);
  });

  it("should render the component", () => {
    render(
      <FallbackSelectionForm
        groups={[]}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByText(/no fallback groups configured/i)).toBeInTheDocument();
  });

  it("should show Create First Group button when no groups exist", () => {
    render(
      <FallbackSelectionForm
        groups={[]}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByRole("button", { name: /create first group/i })).toBeInTheDocument();
  });

  it("should call onGroupsChange when Create First Group is clicked", async () => {
    const user = userEvent.setup();
    render(
      <FallbackSelectionForm
        groups={[]}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );

    await user.click(screen.getByRole("button", { name: /create first group/i }));

    expect(mockOnGroupsChange).toHaveBeenCalledTimes(1);
    const [newGroups] = mockOnGroupsChange.mock.calls[0];
    expect(newGroups).toHaveLength(1);
    expect(newGroups[0]).toEqual({
      id: "1234567890",
      primaryModel: null,
      fallbackModels: [],
    });
  });

  it("should display tabs when groups exist", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: null, fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByRole("tab", { name: /group 1/i })).toBeInTheDocument();
  });

  it("should display primary model as tab label when set", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: "gpt-4", fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByRole("tab", { name: "gpt-4" })).toBeInTheDocument();
  });

  it("should call onGroupsChange when add tab button is clicked", async () => {
    const user = userEvent.setup();
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: null, fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );

    const addTabButton = screen.getByRole("button", { name: /add tab/i });
    await user.click(addTabButton);

    expect(mockOnGroupsChange).toHaveBeenCalledTimes(1);
    const [newGroups] = mockOnGroupsChange.mock.calls[0];
    expect(newGroups).toHaveLength(2);
    expect(newGroups[1]).toEqual({
      id: "1234567890",
      primaryModel: null,
      fallbackModels: [],
    });
  });

  it("should not show add tab button when maxGroups is reached", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: null, fallbackModels: [] },
      { id: "2", primaryModel: null, fallbackModels: [] },
      { id: "3", primaryModel: null, fallbackModels: [] },
      { id: "4", primaryModel: null, fallbackModels: [] },
      { id: "5", primaryModel: null, fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
        maxGroups={5}
      />,
    );
    expect(screen.queryByRole("button", { name: /add tab/i })).not.toBeInTheDocument();
  });

  it("should show add tab button when below maxGroups with custom maxGroups", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: null, fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
        maxGroups={3}
      />,
    );
    expect(screen.getByRole("button", { name: /add tab/i })).toBeInTheDocument();
  });

  it("should call onGroupsChange when a group is removed", async () => {
    const user = userEvent.setup();
    const antd = await import("antd");
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: "gpt-4", fallbackModels: [] },
      { id: "2", primaryModel: "gpt-3.5-turbo", fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );

    const removeButtons = screen.getAllByRole("tab", { name: "remove" });
    await user.click(removeButtons[0]);

    expect(mockOnGroupsChange).toHaveBeenCalledTimes(1);
    const [newGroups] = mockOnGroupsChange.mock.calls[0];
    expect(newGroups).toHaveLength(1);
    expect(newGroups[0].id).toBe("2");
    expect(antd.message.warning).not.toHaveBeenCalled();
  });

  it("should render FallbackGroupConfig for each group", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: null, fallbackModels: [] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByText("Select primary model")).toBeInTheDocument();
    expect(screen.getByText("Primary Model")).toBeInTheDocument();
  });

  it("should display group with primary and fallback models in FallbackGroupConfig", () => {
    const groups: FallbackGroup[] = [
      { id: "1", primaryModel: "gpt-4", fallbackModels: ["gpt-3.5-turbo"] },
    ];
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
      />,
    );
    expect(screen.getByRole("tab", { name: "gpt-4" })).toBeInTheDocument();
    expect(screen.getAllByText("gpt-4").length).toBeGreaterThan(0);
    expect(screen.getByText("gpt-3.5-turbo")).toBeInTheDocument();
  });

  it("should not add group when add button clicked at maxGroups", () => {
    const groups: FallbackGroup[] = Array.from({ length: 5 }, (_, i) => ({
      id: String(i + 1),
      primaryModel: null,
      fallbackModels: [] as string[],
    }));
    render(
      <FallbackSelectionForm
        groups={groups}
        onGroupsChange={mockOnGroupsChange}
        availableModels={AVAILABLE_MODELS}
        maxGroups={5}
      />,
    );

    expect(screen.queryByRole("button", { name: /add tab/i })).not.toBeInTheDocument();
    expect(mockOnGroupsChange).not.toHaveBeenCalled();
  });
});
