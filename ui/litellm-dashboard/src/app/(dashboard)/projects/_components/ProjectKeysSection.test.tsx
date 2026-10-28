import { describe, it, expect, vi, beforeEach, type Mock } from "vitest";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../../../tests/test-utils";
import { ProjectKeysSection } from "./ProjectKeysSection";

const mockUseKeys = vi.fn();
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: (...args: unknown[]) => mockUseKeys(...args),
}));

vi.mock("@/components/common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span>{userId}</span>,
}));

const emptyKeysResponse = {
  data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
  isLoading: false,
};

describe("ProjectKeysSection", () => {
  it("should render", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should show the Keys card title", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByText("Keys")).toBeInTheDocument();
  });

  it("should display the total key count from the API response", () => {
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 42, current_page: 1, total_pages: 9 },
      isLoading: false,
    });
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("of 42");
  });

  it("should show 'No keys found' when the project has no keys", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByText("No keys found")).toBeInTheDocument();
  });

  it("should render a search input for filtering by key name", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByPlaceholderText("Filter by key name...")).toBeInTheDocument();
  });

  it("should call useKeys with the projectId", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-abc" />);
    expect(mockUseKeys).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      expect.objectContaining({ projectID: "proj-abc" }),
    );
  });

  it("should pass null for selectedKeyAlias when the filter input is empty", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(mockUseKeys).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      expect.objectContaining({ selectedKeyAlias: null }),
    );
  });
});

describe("ProjectKeysSection URL state (keys_ prefix)", () => {
  const fortyTwoKeys = {
    data: { keys: [], total_count: 42, current_page: 1, total_pages: 9 },
    isLoading: false,
    isError: false,
  };
  const lastSearchParams = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

  beforeEach(() => {
    mockUseKeys.mockReset();
  });

  it("should fetch the page, page size and key name filter named by the keys_ params", () => {
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, {
      searchParams: "?page=4&keys_page=2&keys_page_size=10&keys_search=prod",
    });

    expect(mockUseKeys).toHaveBeenLastCalledWith(
      2,
      10,
      expect.objectContaining({ projectID: "proj-1", selectedKeyAlias: "prod" }),
    );
    expect(screen.getByPlaceholderText("Filter by key name...")).toHaveValue("prod");
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 5");
  });

  it("should cap an oversized ?keys_page_size= at the largest offered page size", () => {
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, { searchParams: "?keys_page_size=500" });

    expect(mockUseKeys).toHaveBeenLastCalledWith(1, 25, expect.anything());
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("should fall back to the default page size for a ?keys_page_size= outside the offered options", () => {
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, { searchParams: "?keys_page_size=7" });

    expect(mockUseKeys).toHaveBeenLastCalledWith(1, 5, expect.anything());
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 9");
  });

  it("should drop an unsupported ?keys_page_size= when the user pages forward", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, { searchParams: "?keys_page_size=7", onUrlUpdate });

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].queryString).toBe("?keys_page=2"));
    expect(mockUseKeys).toHaveBeenLastCalledWith(2, 5, expect.anything());
  });

  it("should write the key name filter to ?keys_search= and return the keys to their first page", async () => {
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, {
      searchParams: "?page=4&keys_page=3",
      onUrlUpdate,
    });

    fireEvent.change(screen.getByPlaceholderText("Filter by key name..."), { target: { value: "prod" } });

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("keys_search")).toBe("prod"));
    expect(lastSearchParams(onUrlUpdate)?.has("keys_page")).toBe(false);
    expect(lastSearchParams(onUrlUpdate)?.get("page")).toBe("4");
    expect(mockUseKeys).toHaveBeenLastCalledWith(1, 5, expect.objectContaining({ selectedKeyAlias: "prod" }));
  });

  it("should remove ?keys_search= when the key filter is cleared", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, { searchParams: "?keys_search=prod", onUrlUpdate });

    await user.click(screen.getByRole("button", { name: /clear key filter/i }));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.has("keys_search")).toBe(false));
    expect(mockUseKeys).toHaveBeenLastCalledWith(1, 5, expect.objectContaining({ selectedKeyAlias: null }));
  });

  it("should write key pages to ?keys_page= without touching the projects list's ?page=", async () => {
    const user = userEvent.setup();
    mockUseKeys.mockReturnValue(fortyTwoKeys);
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />, { searchParams: "?page=4", onUrlUpdate });

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("keys_page")).toBe("2"));
    expect(lastSearchParams(onUrlUpdate)?.get("page")).toBe("4");
    expect(mockUseKeys).toHaveBeenLastCalledWith(2, 5, expect.anything());
  });

  it("should snap a ?keys_page= past the last page back to the last page once the keys load", async () => {
    mockUseKeys.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { rerender } = renderWithProviders(<ProjectKeysSection projectId="proj-1" />, {
      searchParams: "?keys_page=9",
      onUrlUpdate,
    });
    expect(mockUseKeys).toHaveBeenLastCalledWith(9, 5, expect.anything());

    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 6, current_page: 9, total_pages: 2 },
      isLoading: false,
      isError: false,
    });
    rerender(<ProjectKeysSection projectId="proj-1" />);

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("keys_page")).toBe("2"));
    expect(mockUseKeys).toHaveBeenLastCalledWith(2, 5, expect.anything());
  });

  it("should keep a deep-linked ?keys_page= when the key fetch fails", async () => {
    mockUseKeys.mockReturnValue({ data: undefined, isLoading: true, isError: false });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { rerender } = renderWithProviders(<ProjectKeysSection projectId="proj-1" />, {
      searchParams: "?keys_page=3",
      onUrlUpdate,
    });

    mockUseKeys.mockReturnValue({ data: undefined, isLoading: false, isError: true });
    rerender(<ProjectKeysSection projectId="proj-1" />);

    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(onUrlUpdate).not.toHaveBeenCalled();
    expect(mockUseKeys).toHaveBeenLastCalledWith(3, 5, expect.anything());
  });
});
