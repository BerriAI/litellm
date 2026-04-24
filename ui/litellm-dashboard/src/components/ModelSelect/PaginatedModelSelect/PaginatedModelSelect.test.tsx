import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { PaginatedModelSelect } from "./PaginatedModelSelect";

const mockFetchNextPage = vi.fn();

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useInfiniteModelInfo: vi.fn(),
}));

vi.mock("@tanstack/react-pacer/debouncer", async () => {
  const React = await vi.importActual<typeof import("react")>("react");
  return {
    useDebouncedState: (initial: string) => {
      const [value, setValue] = React.useState(initial);
      return [value, setValue];
    },
  };
});

import { useInfiniteModelInfo } from "@/app/(dashboard)/hooks/models/useModels";

const mockUseInfiniteModelInfo = vi.mocked(useInfiniteModelInfo);

const mockPagesWithModels = {
  pages: [
    {
      data: [
        { model_name: "GPT-4", model_info: { id: "model-1" } },
        { model_name: "Claude-3", model_info: { id: "model-2" } },
      ],
      total_count: 2,
      current_page: 1,
      total_pages: 1,
      size: 50,
    },
  ],
};

const mockEmptyPages = {
  pages: [{ data: [], total_count: 0, current_page: 1, total_pages: 1, size: 50 }],
};

// The popover renders each option as a <button> containing the model name
// and model ID as separate text nodes. Because these are not role="option",
// we locate them via the visible "Model ID: <id>" label.
const getModelOption = (modelId: string) => {
  const dialog = screen.queryByRole("dialog");
  const scope = dialog ? within(dialog) : screen;
  const idText = scope.getByText(`Model ID: ${modelId}`);
  const btn = idText.closest("button");
  if (!btn) throw new Error(`No button for model ${modelId}`);
  return btn;
};
const queryModelOption = (modelId: string) => {
  const dialog = screen.queryByRole("dialog");
  const scope = dialog ? within(dialog) : screen;
  const idText = scope.queryByText(`Model ID: ${modelId}`);
  return idText ? idText.closest("button") : null;
};

describe("PaginatedModelSelect", () => {
  const mockOnChange = vi.fn();

  const defaultHookReturn = {
    data: mockPagesWithModels,
    fetchNextPage: mockFetchNextPage,
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteModelInfo.mockReturnValue(defaultHookReturn as any);
  });

  it("should render", () => {
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByText("Select a model")).toBeInTheDocument();
  });

  it("should display custom placeholder when provided", () => {
    renderWithProviders(
      <PaginatedModelSelect onChange={mockOnChange} placeholder="Choose model" />,
    );

    expect(screen.getByText("Choose model")).toBeInTheDocument();
  });

  it("should display model options when data is loaded", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getModelOption("model-1")).toBeInTheDocument();
      expect(getModelOption("model-2")).toBeInTheDocument();
    });
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText("GPT-4")).toBeInTheDocument();
    expect(within(dialog).getByText("Claude-3")).toBeInTheDocument();
  });

  it("should call onChange when user selects a model", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    const option = await waitFor(() => getModelOption("model-1"));
    await user.click(option);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith("model-1");
    });
  });

  it("should display selected value when value prop is provided", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <PaginatedModelSelect value="model-1" onChange={mockOnChange} />,
    );

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getModelOption("model-1")).toBeInTheDocument();
    });
  });

  it("should show loading state when isLoading is true", () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      isLoading: true,
    } as any);

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    expect(screen.getByRole("combobox")).toHaveAttribute("aria-expanded", "false");
  });

  it("should pass pageSize to useInfiniteModelInfo", () => {
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} pageSize={25} />);

    expect(mockUseInfiniteModelInfo).toHaveBeenCalledWith(25, undefined);
  });

  it("should pass search to useInfiniteModelInfo when user types", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);
    await user.keyboard("gpt");

    await waitFor(() => {
      expect(mockUseInfiniteModelInfo).toHaveBeenCalledWith(50, "gpt");
    });
  });

  it("should have scroll container for infinite loading when hasNextPage is true", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      hasNextPage: true,
      isFetchingNextPage: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    let option: HTMLElement | null = null;
    await waitFor(() => {
      option = getModelOption("model-1");
      expect(option).toBeInTheDocument();
    });

    const scrollableContainer = option!.parentElement;
    expect(scrollableContainer).toHaveClass("overflow-y-auto");
  });

  it("should deduplicate models with same id across pages", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      data: {
        pages: [
          {
            data: [
              { model_name: "GPT-4", model_info: { id: "model-1" } },
              { model_name: "GPT-4 Dupe", model_info: { id: "model-1" } },
            ],
            total_count: 2,
            current_page: 1,
            total_pages: 1,
            size: 50,
          },
        ],
      },
      fetchNextPage: mockFetchNextPage,
      hasNextPage: false,
      isFetchingNextPage: false,
      isLoading: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      const dialog = screen.getByRole("dialog");
      const matches = within(dialog).queryAllByText("Model ID: model-1");
      expect(matches.length).toBe(1);
    });
  });

  it("should skip models without model_info id", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      data: {
        pages: [
          {
            data: [
              { model_name: "Valid Model", model_info: { id: "valid-id" } },
              { model_name: "No ID", model_info: null },
              { model_name: "Empty ID", model_info: { id: "" } },
            ],
            total_count: 3,
            current_page: 1,
            total_pages: 1,
            size: 50,
          },
        ],
      },
      fetchNextPage: mockFetchNextPage,
      hasNextPage: false,
      isFetchingNextPage: false,
      isLoading: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getModelOption("valid-id")).toBeInTheDocument();
    });
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).queryByText("No ID")).not.toBeInTheDocument();
    expect(within(dialog).queryByText("Empty ID")).not.toBeInTheDocument();
  });

  it("should show model ID only when model_name is empty", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      data: {
        pages: [
          {
            data: [{ model_name: "", model_info: { id: "id-only" } }],
            total_count: 1,
            current_page: 1,
            total_pages: 1,
            size: 50,
          },
        ],
      },
      fetchNextPage: mockFetchNextPage,
      hasNextPage: false,
      isFetchingNextPage: false,
      isLoading: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getModelOption("id-only")).toBeInTheDocument();
    });
    // No "Model name:" label is shown when model_name is empty.
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).queryByText("Model name:")).not.toBeInTheDocument();
  });

  it("should respect allowClear prop", () => {
    renderWithProviders(
      <PaginatedModelSelect value="model-1" onChange={mockOnChange} allowClear={false} />,
    );

    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should respect disabled prop", () => {
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} disabled />);

    const combobox = screen.getByRole("combobox");
    expect(combobox).toBeDisabled();
  });

  it("should not call fetchNextPage when hasNextPage is false", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      hasNextPage: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(getModelOption("model-1")).toBeInTheDocument();
    });

    expect(mockFetchNextPage).not.toHaveBeenCalled();
  });
});
