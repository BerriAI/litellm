import React from "react";
import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../tests/test-utils";
import { DocsMenu } from "./DocsMenu";

describe("DocsMenu", () => {
  const items = [
    { label: "Custom pricing", href: "https://docs.example.com/pricing" },
    { label: "Cost tracking", href: "https://docs.example.com/cost" },
  ];

  it("should render the menu button with default text", () => {
    renderWithProviders(<DocsMenu items={items} />);

    expect(screen.getByRole("button", { name: /docs/i })).toBeInTheDocument();
  });

  it("should hide menu items initially", () => {
    renderWithProviders(<DocsMenu items={items} />);
    expect(screen.queryByText("Custom pricing")).not.toBeInTheDocument();
  });

  it("should show menu items when button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DocsMenu items={items} />);

    await user.click(screen.getByRole("button", { name: /docs/i }));

    expect(screen.getByText("Custom pricing")).toBeInTheDocument();
    expect(screen.getByText("Cost tracking")).toBeInTheDocument();
  });

  it("should close the menu when an item is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DocsMenu items={items} />);

    await user.click(screen.getByRole("button", { name: /docs/i }));
    await user.click(screen.getByText("Custom pricing"));

    expect(screen.queryByText("Cost tracking")).not.toBeInTheDocument();
  });

  it("should set aria-expanded correctly based on menu state", async () => {
    const user = userEvent.setup();
    renderWithProviders(<DocsMenu items={items} />);

    const button = screen.getByRole("button", { name: /docs/i });
    expect(button).toHaveAttribute("aria-expanded", "false");

    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("should close menu when clicking outside", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <div>
        <DocsMenu items={items} />
        <button>Outside</button>
      </div>,
    );
    await user.click(screen.getByRole("button", { name: /docs/i }));
    expect(screen.getByText("Custom pricing")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /outside/i }));
    expect(screen.queryByText("Custom pricing")).not.toBeInTheDocument();
  });
});
