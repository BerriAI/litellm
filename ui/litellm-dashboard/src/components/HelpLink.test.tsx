import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HelpLink, HelpIcon, DocsMenu } from "./HelpLink";

describe("HelpLink", () => {
  it("should render", () => {
    render(<HelpLink href="https://docs.example.com" />);
    expect(screen.getByRole("link")).toBeInTheDocument();
  });

  it("should display default 'Learn more' text when no children provided", () => {
    render(<HelpLink href="https://docs.example.com" />);
    expect(screen.getByText("Learn more")).toBeInTheDocument();
  });

  it("should display custom children text", () => {
    render(<HelpLink href="https://docs.example.com">Custom docs link</HelpLink>);
    expect(screen.getByText("Custom docs link")).toBeInTheDocument();
  });

  it("should open in a new tab with noopener noreferrer", () => {
    render(<HelpLink href="https://docs.example.com" />);
    const link = screen.getByRole("link");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("should have accessible screen reader text", () => {
    render(<HelpLink href="https://docs.example.com" />);
    expect(screen.getByText("(opens in a new tab)")).toBeInTheDocument();
  });
});

describe("HelpIcon", () => {
  it("should render", () => {
    render(<HelpIcon content="Some help text" />);
    expect(screen.getByRole("button", { name: /help information/i })).toBeInTheDocument();
  });

  it("should show tooltip content on mouse enter", async () => {
    const user = userEvent.setup();
    render(<HelpIcon content="Helpful tooltip text" />);
    await user.hover(screen.getByRole("button", { name: /help information/i }));
    expect(screen.getByText("Helpful tooltip text")).toBeInTheDocument();
  });

  it("should hide tooltip content on mouse leave", async () => {
    const user = userEvent.setup();
    render(<HelpIcon content="Helpful tooltip text" />);
    const button = screen.getByRole("button", { name: /help information/i });
    await user.hover(button);
    await user.unhover(button);
    expect(screen.queryByText("Helpful tooltip text")).not.toBeInTheDocument();
  });

  it("should show learn more link when learnMoreHref is provided", async () => {
    const user = userEvent.setup();
    render(<HelpIcon content="Help text" learnMoreHref="https://docs.example.com" />);
    await user.hover(screen.getByRole("button", { name: /help information/i }));
    expect(screen.getByText("Learn more")).toBeInTheDocument();
  });

  it("should use custom learnMoreText when provided", async () => {
    const user = userEvent.setup();
    render(
      <HelpIcon content="Help text" learnMoreHref="https://docs.example.com" learnMoreText="Read docs" />,
    );
    await user.hover(screen.getByRole("button", { name: /help information/i }));
    expect(screen.getByText("Read docs")).toBeInTheDocument();
  });
});

describe("DocsMenu", () => {
  const items = [
    { label: "Custom pricing", href: "https://docs.example.com/pricing" },
    { label: "Spend tracking", href: "https://docs.example.com/spend" },
  ];

  it("should render", () => {
    render(<DocsMenu items={items} />);
    expect(screen.getByRole("button", { name: /docs/i })).toBeInTheDocument();
  });

  it("should show menu items when clicked", async () => {
    const user = userEvent.setup();
    render(<DocsMenu items={items} />);
    await user.click(screen.getByRole("button", { name: /docs/i }));
    expect(screen.getByText("Custom pricing")).toBeInTheDocument();
    expect(screen.getByText("Spend tracking")).toBeInTheDocument();
  });

  it("should hide menu items when clicked again", async () => {
    const user = userEvent.setup();
    render(<DocsMenu items={items} />);
    const button = screen.getByRole("button", { name: /docs/i });
    await user.click(button);
    await user.click(button);
    expect(screen.queryByText("Custom pricing")).not.toBeInTheDocument();
  });

  it("should set aria-expanded correctly", async () => {
    const user = userEvent.setup();
    render(<DocsMenu items={items} />);
    const button = screen.getByRole("button", { name: /docs/i });
    expect(button).toHaveAttribute("aria-expanded", "false");
    await user.click(button);
    expect(button).toHaveAttribute("aria-expanded", "true");
  });

  it("should close menu when clicking outside", async () => {
    const user = userEvent.setup();
    render(
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

  it("should display custom children text", () => {
    render(<DocsMenu items={items}>Help</DocsMenu>);
    expect(screen.getByText("Help")).toBeInTheDocument();
  });
});
