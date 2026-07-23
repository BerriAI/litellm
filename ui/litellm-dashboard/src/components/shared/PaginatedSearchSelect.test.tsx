import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PaginatedSearchSelect } from "./PaginatedSearchSelect";
import type { SearchSelectOption } from "./SearchSelect";

const OPTIONS: SearchSelectOption[] = [
  { label: "alias-alpha", value: "alias-alpha" },
  { label: "alias-beta", value: "alias-beta" },
  { label: "gamma-key", value: "gamma-key" },
];

function renderSelect(overrides: Partial<React.ComponentProps<typeof PaginatedSearchSelect>> = {}) {
  const props: React.ComponentProps<typeof PaginatedSearchSelect> = {
    options: OPTIONS,
    onValueChange: vi.fn(),
    onSearchChange: vi.fn(),
    onLoadMore: vi.fn(),
    ...overrides,
  };
  render(<PaginatedSearchSelect {...props} />);
  return props;
}

function setListMetrics(list: HTMLElement, metrics: { scrollTop: number; clientHeight: number; scrollHeight: number }) {
  Object.defineProperty(list, "scrollTop", { value: metrics.scrollTop, configurable: true });
  Object.defineProperty(list, "clientHeight", { value: metrics.clientHeight, configurable: true });
  Object.defineProperty(list, "scrollHeight", { value: metrics.scrollHeight, configurable: true });
}

describe("PaginatedSearchSelect", () => {
  it("reports the typed query to the server instead of filtering locally", async () => {
    const user = userEvent.setup();
    const onSearchChange = vi.fn();
    renderSelect({ onSearchChange });

    const input = screen.getByRole("combobox");
    await user.click(input);
    await user.type(input, "gamma");

    await waitFor(() => expect(onSearchChange).toHaveBeenCalledWith("gamma"));

    expect(await screen.findByText("alias-alpha")).toBeInTheDocument();
  });

  it("requests the next page once the list is scrolled near the bottom", async () => {
    const user = userEvent.setup();
    const onLoadMore = vi.fn();
    renderSelect({ onLoadMore, hasNextPage: true });

    await user.click(screen.getByRole("combobox"));
    const list = await screen.findByTestId("paginated-search-select-list");

    setListMetrics(list, { scrollTop: 0, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).not.toHaveBeenCalled();

    setListMetrics(list, { scrollTop: 850, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("does not request more pages when there is no next page or one is already in flight", async () => {
    const user = userEvent.setup();
    const onLoadMore = vi.fn();
    const { unmount } = render(
      <PaginatedSearchSelect
        options={OPTIONS}
        onValueChange={vi.fn()}
        onSearchChange={vi.fn()}
        onLoadMore={onLoadMore}
        hasNextPage={false}
      />,
    );
    await user.click(screen.getByRole("combobox"));
    let list = await screen.findByTestId("paginated-search-select-list");
    setListMetrics(list, { scrollTop: 900, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).not.toHaveBeenCalled();
    unmount();

    renderSelect({ onLoadMore, hasNextPage: true, isFetchingNextPage: true });
    await user.click(screen.getByRole("combobox"));
    list = await screen.findByTestId("paginated-search-select-list");
    setListMetrics(list, { scrollTop: 900, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).not.toHaveBeenCalled();
  });

  it("keeps showing a selected value that is absent from the current page of options", () => {
    renderSelect({ options: [], value: "alias-from-an-earlier-page" });

    expect(screen.getByRole("combobox")).toHaveValue("alias-from-an-earlier-page");
  });

  it("reports the selected option's value", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    renderSelect({ onValueChange });

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("alias-beta"));

    expect(onValueChange).toHaveBeenCalledWith("alias-beta");
  });

  it("surfaces loading and fetching-more affordances", async () => {
    const user = userEvent.setup();
    const { unmount } = render(
      <PaginatedSearchSelect
        options={[]}
        onValueChange={vi.fn()}
        onSearchChange={vi.fn()}
        onLoadMore={vi.fn()}
        isLoading
        loadingText="Loading key aliases…"
      />,
    );
    await user.click(screen.getByRole("combobox"));
    expect(await screen.findByText("Loading key aliases…")).toBeInTheDocument();
    unmount();

    renderSelect({ isFetchingNextPage: true });
    await user.click(screen.getByRole("combobox"));
    expect(await screen.findByTestId("paginated-search-select-loading-more")).toBeInTheDocument();
  });
});
