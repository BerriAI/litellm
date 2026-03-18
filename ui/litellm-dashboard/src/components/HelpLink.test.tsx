import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../tests/test-utils";
import { HelpLink, HelpIcon, DocsMenu } from "./HelpLink";

describe("HelpLink", () => {
  it("should render with default children and open in new tab", () => {
    renderWithProviders(<HelpLink href="https://docs.example.com" />);

    const link = screen.getByRole("link", { name: /learn more/i });
    expect(link).toHaveAttribute("href", "https://docs.example.com");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should render custom children text", () => {
    renderWithProviders(
      <HelpLink href="https://docs.example.com">Custom docs link</HelpLink>
    );

    expect(screen.getByText("Custom docs link")).toBeInTheDocument();
  });

  it("should include a screen-reader-only label for accessibility", () => {
    renderWithProviders(<HelpLink href="https://docs.example.com" />);

    expect(screen.getByText("(opens in a new tab)")).toBeInTheDocument();
  });
});

describe("HelpIcon", () => {
  it("should render a help button with accessible label", () => {
    renderWithProviders(<HelpIcon content="Some help text" />);

    expect(screen.getByRole("button", { name: /help information/i })).toBeInTheDocument();
  });

  it("should show tooltip content on hover", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HelpIcon content="Tooltip help text" />);

    await user.hover(screen.getByRole("button", { name: /help information/i }));

    expect(screen.getByText("Tooltip help text")).toBeInTheDocument();
  });

  it("should show learn more link when learnMoreHref is provided", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <HelpIcon
        content="Help text"
        learnMoreHref="https://docs.example.com"
        learnMoreText="Read docs"
      />
    );

    await user.hover(screen.getByRole("button", { name: /help information/i }));

    const link = screen.getByRole("link", { name: /read docs/i });
    expect(link).toHaveAttribute("href", "https://docs.example.com");
  });

  it("should not show learn more link when learnMoreHref is not provided", async () => {
    const user = userEvent.setup();
    renderWithProviders(<HelpIcon content="Help text" />);

    await user.hover(screen.getByRole("button", { name: /help information/i }));

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});

describe("DocsMenu", () => {
  const items = [
    { label: "Custom pricing", href: "https://docs.example.com/pricing" },
    { label: "Cost tracking", href: "https://docs.example.com/cost" },
  ];

  it("should render the menu button with default text", () => {
    renderWithProviders(<DocsMenu items={items} />);

    expect(screen.getByRole("button", { name: /docs/i })).toBeInTheDocument();
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
