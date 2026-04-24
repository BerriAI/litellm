import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "../../../../tests/test-utils";
import { PaginatedKeyAliasSelect } from "./PaginatedKeyAliasSelect";

const mockFetchNextPage = vi.fn();

vi.mock("@/app/(dashboard)/hooks/keys/useKeyAliases", () => ({
  useInfiniteKeyAliases: vi.fn(),
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

import { useInfiniteKeyAliases } from "@/app/(dashboard)/hooks/keys/useKeyAliases";

const mockUseInfiniteKeyAliases = vi.mocked(useInfiniteKeyAliases);

const mockPagesWithAliases = {
  pages: [
    {
      aliases: ["alias-1", "alias-2"],
      total_count: 2,
      current_page: 1,
      total_pages: 1,
      size: 50,
    },
  ],
};

const mockEmptyPages = {
  pages: [{ aliases: [], total_count: 0, current_page: 1, total_pages: 1, size: 50 }],
};

// The popover renders option rows as plain <button> elements inside the
// popover content. This helper fetches them scoped to the open popover.
const getAliasButton = (name: string) => {
  const dialog = screen.queryByRole("dialog");
  const scope = dialog ? within(dialog) : screen;
  return scope.getByRole("button", { name });
};
const queryAllAliasButtons = (name: string) => {
  const dialog = screen.queryByRole("dialog");
  const scope = dialog ? within(dialog) : screen;
  return scope.queryAllByRole("button", { name });
};

describe("PaginatedKeyAliasSelect", () => {
  const mockOnChange = vi.fn();

  const defaultHookReturn = {
    data: mockPagesWithAliases,
    fetchNextPage: mockFetchNextPage,
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseInfiniteKeyAliases.mockReturnValue(defaultHookReturn as any);
  });

  it("should render", () => {
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    expect(screen.getByRole("combobox")).toBeInTheDocument();
    expect(screen.getByText("Select a key alias")).toBeInTheDocument();
  });

  it("should display custom placeholder when provided", () => {
    renderWithProviders(
      <PaginatedKeyAliasSelect onChange={mockOnChange} placeholder="Choose alias" />,
    );

    expect(screen.getByText("Choose alias")).toBeInTheDocument();
  });

  it("should display alias options when data is loaded", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getAliasButton("alias-1")).toBeInTheDocument();
      expect(getAliasButton("alias-2")).toBeInTheDocument();
    });
  });

  it("should call onChange when user selects an alias", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    const option = await waitFor(() => getAliasButton("alias-1"));
    await user.click(option);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith("alias-1");
    });
  });

  it("should show loading state when isLoading is true", () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      isLoading: true,
    } as any);

    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    expect(screen.getByRole("combobox")).toHaveAttribute("aria-expanded", "false");
  });

  it("should pass pageSize to useInfiniteKeyAliases", () => {
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} pageSize={25} />);

    expect(mockUseInfiniteKeyAliases).toHaveBeenCalledWith(25, undefined, undefined);
  });

  it("should pass search to useInfiniteKeyAliases when user types", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);
    await user.keyboard("my-alias");

    await waitFor(() => {
      expect(mockUseInfiniteKeyAliases).toHaveBeenCalledWith(50, "my-alias", undefined);
    });
  });

  it("should have scroll container for infinite loading when hasNextPage is true", async () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      hasNextPage: true,
      isFetchingNextPage: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getAliasButton("alias-1")).toBeInTheDocument();
    });

    // The scrollable area wraps the option buttons inside the popover content.
    const scrollableContainer = getAliasButton("alias-1").parentElement;
    expect(scrollableContainer).toHaveClass("overflow-y-auto");
  });

  it("should deduplicate aliases with the same value across pages", async () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      data: {
        pages: [
          {
            aliases: ["alias-1", "alias-1"],
            total_count: 2,
            current_page: 1,
            total_pages: 1,
            size: 50,
          },
        ],
      },
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      const options = queryAllAliasButtons("alias-1");
      expect(options.length).toBe(1);
    });
  });

  it("should skip empty aliases", async () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      data: {
        pages: [
          {
            aliases: ["valid-alias", "", null],
            total_count: 3,
            current_page: 1,
            total_pages: 1,
            size: 50,
          },
        ],
      },
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(getAliasButton("valid-alias")).toBeInTheDocument();
    });
    // Only the single valid-alias row is rendered inside the popover.
    const dialog = screen.getByRole("dialog");
    const rows = within(dialog).getAllByRole("button");
    // Rows = option button(s); the trigger button is outside the dialog.
    expect(rows.length).toBe(1);
  });

  it("should respect allowClear prop", () => {
    renderWithProviders(
      <PaginatedKeyAliasSelect value="alias-1" onChange={mockOnChange} allowClear={false} />,
    );

    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should respect disabled prop", () => {
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} disabled />);

    const combobox = screen.getByRole("combobox");
    expect(combobox).toBeDisabled();
  });

  it("should not call fetchNextPage when hasNextPage is false", async () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      hasNextPage: false,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    await user.click(screen.getByRole("combobox"));

    await waitFor(() => {
      expect(getAliasButton("alias-1")).toBeInTheDocument();
    });

    expect(mockFetchNextPage).not.toHaveBeenCalled();
  });

  it("should show no aliases found when data is empty", async () => {
    mockUseInfiniteKeyAliases.mockReturnValue({
      ...defaultHookReturn,
      data: mockEmptyPages,
    } as any);

    const user = userEvent.setup();
    renderWithProviders(<PaginatedKeyAliasSelect onChange={mockOnChange} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    await waitFor(() => {
      expect(screen.getByText("No key aliases found")).toBeInTheDocument();
    });
  });
});
