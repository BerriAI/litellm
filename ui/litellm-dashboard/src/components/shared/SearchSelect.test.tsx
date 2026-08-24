import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SearchSelect } from "./SearchSelect";

const OPTIONS = [
  { label: "Acme Prod", value: "team-1" },
  { label: "Growth", value: "team-2" },
  { label: "Data Team", value: "team-3" },
];

describe("SearchSelect", () => {
  it("renders the placeholder when nothing is selected", () => {
    render(<SearchSelect options={OPTIONS} onValueChange={vi.fn()} placeholder="Select Team…" />);
    expect(screen.getByPlaceholderText("Select Team…")).toBeInTheDocument();
  });

  it("names the field from aria-label so callers can label it independently of the placeholder", () => {
    render(
      <SearchSelect options={OPTIONS} onValueChange={vi.fn()} placeholder="Select Team…" aria-label="Default model" />,
    );
    expect(screen.getByRole("combobox", { name: "Default model" })).toBeInTheDocument();
  });

  it("shows the selected option's label in the field", () => {
    render(<SearchSelect options={OPTIONS} value="team-2" onValueChange={vi.fn()} />);
    expect(screen.getByRole("combobox")).toHaveValue("Growth");
  });

  it("shows a value the options do not carry yet instead of blanking the field", () => {
    const { rerender } = render(<SearchSelect options={[]} value="team-2" onValueChange={vi.fn()} />);
    expect(screen.getByRole("combobox")).toHaveValue("team-2");
    rerender(<SearchSelect options={OPTIONS} value="team-2" onValueChange={vi.fn()} />);
    expect(screen.getByRole("combobox")).toHaveValue("Growth");
  });

  it("shows a clear control only when a value is selected", () => {
    const { rerender } = render(<SearchSelect options={OPTIONS} onValueChange={vi.fn()} />);
    expect(document.querySelector('[data-slot="combobox-clear"]')).toBeNull();
    rerender(<SearchSelect options={OPTIONS} value="team-1" onValueChange={vi.fn()} />);
    expect(document.querySelector('[data-slot="combobox-clear"]')).not.toBeNull();
  });

  it("filters the options client-side as you type", async () => {
    const user = userEvent.setup();
    render(<SearchSelect options={OPTIONS} onValueChange={vi.fn()} />);
    const input = screen.getByRole("combobox");
    await user.click(input);
    fireEvent.change(input, { target: { value: "grow" } });
    expect(await screen.findByText("Growth")).toBeInTheDocument();
    expect(screen.queryByText("Acme Prod")).not.toBeInTheDocument();
  });

  it("renders a muted sublabel and matches it when searching", async () => {
    const user = userEvent.setup();
    render(
      <SearchSelect
        options={[{ label: "Acme Prod", value: "team-1", sublabel: "team-abc-123" }]}
        onValueChange={vi.fn()}
      />,
    );
    const input = screen.getByRole("combobox");
    await user.click(input);
    expect(await screen.findByText("team-abc-123")).toBeInTheDocument();
    fireEvent.change(input, { target: { value: "abc-123" } });
    expect(await screen.findByText("Acme Prod")).toBeInTheDocument();
  });

  it("selects an option and reports its value", async () => {
    const onValueChange = vi.fn();
    const user = userEvent.setup();
    render(<SearchSelect options={OPTIONS} onValueChange={onValueChange} />);
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Growth"));
    expect(onValueChange).toHaveBeenCalledWith("team-2");
  });

  it("makes the field non-interactive when disabled", async () => {
    const onValueChange = vi.fn();
    const user = userEvent.setup();
    render(<SearchSelect options={OPTIONS} value="team-1" onValueChange={onValueChange} disabled />);

    const input = screen.getByRole("combobox");
    expect(input).toBeDisabled();

    await user.click(input);
    await user.keyboard("Growth");

    expect(input).toHaveValue("Acme Prod");
    expect(screen.queryByText("Growth")).not.toBeInTheDocument();
    expect(onValueChange).not.toHaveBeenCalled();
  });
});
