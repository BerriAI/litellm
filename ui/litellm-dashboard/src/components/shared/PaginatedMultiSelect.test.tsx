import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { PaginatedMultiSelect } from "./PaginatedMultiSelect";
import type { SearchSelectOption } from "./SearchSelect";

const OPTIONS: SearchSelectOption[] = [
  { label: "alias-alpha", value: "alias-alpha" },
  { label: "alias-beta", value: "alias-beta" },
  { label: "gamma-key", value: "gamma-key" },
];

function renderSelect(overrides: Partial<React.ComponentProps<typeof PaginatedMultiSelect>> = {}) {
  const props: React.ComponentProps<typeof PaginatedMultiSelect> = {
    options: OPTIONS,
    onValueChange: vi.fn(),
    onSearchChange: vi.fn(),
    onLoadMore: vi.fn(),
    ...overrides,
  };
  render(<PaginatedMultiSelect {...props} />);
  return props;
}

function setListMetrics(list: HTMLElement, metrics: { scrollTop: number; clientHeight: number; scrollHeight: number }) {
  Object.defineProperty(list, "scrollTop", { value: metrics.scrollTop, configurable: true });
  Object.defineProperty(list, "clientHeight", { value: metrics.clientHeight, configurable: true });
  Object.defineProperty(list, "scrollHeight", { value: metrics.scrollHeight, configurable: true });
}

function chipRemoveButton(label: string): HTMLElement {
  const chip = screen.getByText(label).closest('[data-slot="combobox-chip"]');
  if (chip === null) throw new Error(`no chip found for ${label}`);
  const button = chip.querySelector('[data-slot="combobox-chip-remove"]');
  if (button === null) throw new Error(`no remove control found on chip for ${label}`);
  return button as HTMLElement;
}

describe("PaginatedMultiSelect", () => {
  it("reports the cleared query upstream after a selection, so the next open is not still filtered", async () => {
    const user = userEvent.setup();
    const onSearchChange = vi.fn();

    function Controlled() {
      const [value, setValue] = useState<string[]>([]);
      return (
        <PaginatedMultiSelect
          options={OPTIONS}
          value={value}
          onValueChange={setValue}
          onSearchChange={onSearchChange}
          onLoadMore={vi.fn()}
        />
      );
    }
    render(<Controlled />);

    const input = screen.getByRole("combobox");
    await user.click(input);
    await user.type(input, "alias-a");
    await waitFor(() => expect(onSearchChange).toHaveBeenCalledWith("alias-a"), { timeout: 2000 });

    const list = await screen.findByTestId("paginated-multi-select-list");
    await user.click(within(list).getByText("alias-alpha"));

    expect(input).toHaveValue("");
    await waitFor(() => expect(onSearchChange).toHaveBeenLastCalledWith(""), { timeout: 2000 });
  });

  it("selects multiple values and reports them cumulatively", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    function Controlled() {
      const [value, setValue] = useState<string[]>([]);
      return (
        <PaginatedMultiSelect
          options={OPTIONS}
          value={value}
          onValueChange={(next) => {
            setValue(next);
            onValueChange(next);
          }}
          onSearchChange={vi.fn()}
          onLoadMore={vi.fn()}
        />
      );
    }
    render(<Controlled />);

    const input = screen.getByRole("combobox");
    await user.click(input);
    const list = await screen.findByTestId("paginated-multi-select-list");
    await user.click(within(list).getByText("alias-alpha"));

    await user.click(input);
    await user.click(within(list).getByText("gamma-key"));

    expect(onValueChange).toHaveBeenLastCalledWith(["alias-alpha", "gamma-key"]);
    const chips = document.querySelector('[data-slot="combobox-chips"]') as HTMLElement;
    expect(within(chips).getByText("alias-alpha")).toBeInTheDocument();
    expect(within(chips).getByText("gamma-key")).toBeInTheDocument();
  });

  it("deselects one value via the chip remove control and keeps the rest", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();

    function Controlled() {
      const [value, setValue] = useState<string[]>(["alias-alpha", "alias-beta"]);
      return (
        <PaginatedMultiSelect
          options={OPTIONS}
          value={value}
          onValueChange={(next) => {
            setValue(next);
            onValueChange(next);
          }}
          onSearchChange={vi.fn()}
          onLoadMore={vi.fn()}
        />
      );
    }
    render(<Controlled />);

    await user.click(chipRemoveButton("alias-alpha"));

    expect(onValueChange).toHaveBeenCalledWith(["alias-beta"]);
    expect(screen.queryByText("alias-alpha")).not.toBeInTheDocument();
    expect(screen.getByText("alias-beta")).toBeInTheDocument();
  });

  it("keeps a selected chip visible after the options page no longer contains it", () => {
    const { rerender } = render(
      <PaginatedMultiSelect
        options={OPTIONS}
        value={["ghost-key"]}
        onValueChange={vi.fn()}
        onSearchChange={vi.fn()}
        onLoadMore={vi.fn()}
      />,
    );

    expect(screen.getByText("ghost-key")).toBeInTheDocument();

    rerender(
      <PaginatedMultiSelect
        options={[{ label: "alias-beta", value: "alias-beta" }]}
        value={["ghost-key"]}
        onValueChange={vi.fn()}
        onSearchChange={vi.fn()}
        onLoadMore={vi.fn()}
      />,
    );

    expect(screen.getByText("ghost-key")).toBeInTheDocument();
  });

  it("keeps a picked chip's label after the search filters it off the options page", async () => {
    const user = userEvent.setup();

    const aliased: SearchSelectOption[] = [
      { label: "Prod Alpha", value: "hash-alpha" },
      { label: "Staging Beta", value: "hash-beta" },
    ];

    function Controlled({ options }: { options: SearchSelectOption[] }) {
      const [value, setValue] = useState<string[]>([]);
      return (
        <PaginatedMultiSelect
          options={options}
          value={value}
          onValueChange={setValue}
          onSearchChange={vi.fn()}
          onLoadMore={vi.fn()}
        />
      );
    }
    const { rerender } = render(<Controlled options={aliased} />);

    const input = screen.getByRole("combobox");
    await user.click(input);
    const list = await screen.findByTestId("paginated-multi-select-list");
    await user.click(within(list).getByText("Prod Alpha"));

    rerender(<Controlled options={[{ label: "Staging Beta", value: "hash-beta" }]} />);

    const chips = document.querySelector('[data-slot="combobox-chips"]') as HTMLElement;
    expect(within(chips).getByText("Prod Alpha")).toBeInTheDocument();
    expect(within(chips).queryByText("hash-alpha")).not.toBeInTheDocument();
  });

  it("anchors the dropdown to the chips container so it tracks the growing chip box", async () => {
    const user = userEvent.setup();
    renderSelect({});

    await user.click(screen.getByRole("combobox"));
    await screen.findByTestId("paginated-multi-select-list");

    const content = document.querySelector('[data-slot="combobox-content"]');
    expect(content).toHaveAttribute("data-chips", "true");
  });

  it("does not request the next page on scroll when there is no next page", async () => {
    const user = userEvent.setup();
    const onLoadMore = vi.fn();
    renderSelect({ onLoadMore, hasNextPage: false });

    const input = screen.getByRole("combobox");
    await user.click(input);
    const list = await screen.findByTestId("paginated-multi-select-list");

    setListMetrics(list, { scrollTop: 900, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);

    expect(onLoadMore).not.toHaveBeenCalled();
  });

  it("requests the next page once scrolled past the threshold when a next page exists", async () => {
    const user = userEvent.setup();
    const onLoadMore = vi.fn();
    renderSelect({ onLoadMore, hasNextPage: true });

    const input = screen.getByRole("combobox");
    await user.click(input);
    const list = await screen.findByTestId("paginated-multi-select-list");

    setListMetrics(list, { scrollTop: 0, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).not.toHaveBeenCalled();

    setListMetrics(list, { scrollTop: 850, clientHeight: 100, scrollHeight: 1000 });
    fireEvent.scroll(list);
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });
});
