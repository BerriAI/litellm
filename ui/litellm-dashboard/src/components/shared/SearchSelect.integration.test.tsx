import { fireEvent, renderWithProviders as render, screen } from "../../../tests/test-utils";
import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { SearchSelect } from "./SearchSelect";
import { chooseSelectOption } from "../../../tests/test-utils";

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

  it("should clear to null and allow selecting again through the real control", async () => {
    const onValueChange = vi.fn();
    const user = userEvent.setup();
    function Controlled() {
      const [value, setValue] = useState<string | null>(null);
      return (
        <SearchSelect
          options={OPTIONS}
          value={value}
          onValueChange={(next) => {
            setValue(next);
            onValueChange(next);
          }}
        />
      );
    }
    render(<Controlled />);
    expect(screen.queryByRole("button", { name: "Clear" })).not.toBeInTheDocument();
    await chooseSelectOption(user, screen.getByRole("combobox"), "Growth");
    await user.click(screen.getByRole("button", { name: "Clear" }));
    expect(onValueChange).toHaveBeenLastCalledWith(null);
    expect(screen.getByRole("combobox")).toHaveValue("");
    await chooseSelectOption(user, screen.getByRole("combobox"), "Data Team");
    expect(onValueChange).toHaveBeenLastCalledWith("team-3");
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
    await chooseSelectOption(user, screen.getByRole("combobox"), "Growth");
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
