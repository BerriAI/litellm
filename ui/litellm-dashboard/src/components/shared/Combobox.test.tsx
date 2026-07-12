import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Combobox } from "./Combobox";

const OPTIONS = [
  { label: "Acme Prod", value: "team-1" },
  { label: "Growth", value: "team-2" },
  { label: "Data Team", value: "team-3" },
];

describe("Combobox", () => {
  it("shows the placeholder until a value is selected", () => {
    render(<Combobox options={OPTIONS} onValueChange={vi.fn()} placeholder="Select Team…" />);
    expect(screen.getByRole("combobox")).toHaveTextContent("Select Team…");
  });

  it("shows the selected option's label", () => {
    render(<Combobox options={OPTIONS} value="team-2" onValueChange={vi.fn()} />);
    expect(screen.getByRole("combobox")).toHaveTextContent("Growth");
  });

  it("opens, filters client-side, and selects an option", async () => {
    const onValueChange = vi.fn();
    const user = userEvent.setup();
    render(<Combobox options={OPTIONS} onValueChange={onValueChange} searchPlaceholder="Search teams…" />);

    await user.click(screen.getByRole("combobox"));
    await user.type(await screen.findByPlaceholderText("Search teams…"), "grow");

    expect(screen.getByText("Growth")).toBeInTheDocument();
    expect(screen.queryByText("Acme Prod")).not.toBeInTheDocument();

    await user.click(screen.getByText("Growth"));
    expect(onValueChange).toHaveBeenCalledWith("team-2");
  });

  it("clears the selection", async () => {
    const onValueChange = vi.fn();
    const user = userEvent.setup();
    render(<Combobox options={OPTIONS} value="team-1" onValueChange={onValueChange} />);
    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Clear selection"));
    expect(onValueChange).toHaveBeenCalledWith("");
  });

  it("delegates search to the parent in server-search mode and keeps all options", async () => {
    const onSearchChange = vi.fn();
    const user = userEvent.setup();
    render(
      <Combobox
        options={OPTIONS}
        onValueChange={vi.fn()}
        searchValue=""
        onSearchChange={onSearchChange}
        searchPlaceholder="Search…"
      />,
    );
    await user.click(screen.getByRole("combobox"));
    await user.type(await screen.findByPlaceholderText("Search…"), "x");
    expect(onSearchChange).toHaveBeenCalledWith("x");
    expect(screen.getByText("Acme Prod")).toBeInTheDocument();
  });
});
