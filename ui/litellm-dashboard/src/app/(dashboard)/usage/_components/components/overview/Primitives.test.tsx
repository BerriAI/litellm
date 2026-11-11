import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React, { useState } from "react";
import { describe, expect, it } from "vitest";
import { Segmented } from "./Primitives";

const OPTIONS = [
  { value: "spend", label: "Spend" },
  { value: "tokens", label: "Tokens" },
  { value: "requests", label: "Requests" },
] as const;

function Harness() {
  const [value, setValue] = useState<(typeof OPTIONS)[number]["value"]>("spend");
  return <Segmented label="Metric" value={value} options={OPTIONS} onChange={setValue} />;
}

describe("Segmented", () => {
  it("is a radio group with one tab stop that moves selection with the arrow keys", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    expect(screen.getByRole("radiogroup", { name: "Metric" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "Spend" })).toBeChecked();

    await user.tab();
    expect(screen.getByRole("radio", { name: "Spend" })).toHaveFocus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("radio", { name: "Tokens" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Tokens" })).toHaveFocus();

    // A single tab stop: tabbing again leaves the group rather than visiting each option.
    await user.tab();
    expect(screen.getByRole("radio", { name: "Requests" })).not.toHaveFocus();
  });

  it("selects an option on click", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    await user.click(screen.getByRole("radio", { name: "Requests" }));
    expect(screen.getByRole("radio", { name: "Requests" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Spend" })).not.toBeChecked();
  });
});
