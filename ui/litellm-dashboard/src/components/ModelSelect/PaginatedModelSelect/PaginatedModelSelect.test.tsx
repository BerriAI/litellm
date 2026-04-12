import { screen, waitFor } from "@testing-library/react";
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
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "GPT-4 (model-1)" })).toBeInTheDocument();
      expect(screen.getByRole("option", { name: "Claude-3 (model-2)" })).toBeInTheDocument();
    });
  });

  it("should call onChange when user selects a model", async () => {
    const user = userEvent.setup({ delay: null });
    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    const visibleOption = await screen.findByTitle("GPT-4 (model-1)");
    await user.click(visibleOption);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith("model-1");
    });
  });

  it("should display selected value when value prop is provided", async () => {
    renderWithProviders(
      <PaginatedModelSelect value="model-1" onChange={mockOnChange} />,
    );

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "GPT-4 (model-1)" })).toBeInTheDocument();
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

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "GPT-4 (model-1)" })).toBeInTheDocument();
    });

    const scrollableContainer = document.querySelector(
      ".ant-select-dropdown .rc-virtual-list-holder",
    );
    expect(scrollableContainer).toBeInTheDocument();
    expect(scrollableContainer).toHaveAttribute("style");
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

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      const model1Options = screen.queryAllByRole("option", { name: /model-1/ });
      expect(model1Options.length).toBe(1);
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

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "Valid Model (valid-id)" })).toBeInTheDocument();
      expect(screen.queryByRole("option", { name: "No ID" })).not.toBeInTheDocument();
      expect(screen.queryByRole("option", { name: "Empty ID" })).not.toBeInTheDocument();
    });
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

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await userEvent.click(combobox);

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "id-only" })).toBeInTheDocument();
    });
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
    expect(combobox.closest(".ant-select")).toHaveClass("ant-select-disabled");
  });

  it("should not call fetchNextPage when hasNextPage is false", async () => {
    mockUseInfiniteModelInfo.mockReturnValue({
      ...defaultHookReturn,
      hasNextPage: false,
    } as any);

    renderWithProviders(<PaginatedModelSelect onChange={mockOnChange} />);

    await userEvent.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(screen.getByRole("option", { name: "GPT-4 (model-1)" })).toBeInTheDocument();
    });

    expect(mockFetchNextPage).not.toHaveBeenCalled();
  });
});
