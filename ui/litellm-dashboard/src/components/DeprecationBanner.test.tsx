import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { DeprecationBanner } from "./DeprecationBanner";

describe("DeprecationBanner", () => {
  it("names the deprecated feature in the heading and the body", () => {
    render(<DeprecationBanner featureName="Memory" />);

    expect(screen.getByText("Memory is on a draft deprecation list")).toBeInTheDocument();
    expect(screen.getByText(/Memory is one of several experimental features/)).toBeInTheDocument();
  });

  it("states the target removal date and that the list is not final", () => {
    render(<DeprecationBanner featureName="Memory" />);

    expect(screen.getByText(/as early as September 1, 2026/)).toBeInTheDocument();
    expect(screen.getByText(/This list is a draft and is not final/)).toBeInTheDocument();
  });

  it("links to the deprecation discussion in a new tab without leaking the opener", () => {
    render(<DeprecationBanner featureName="Memory" />);

    const link = screen.getByRole("link", { name: "deprecation discussion" });
    expect(link).toHaveAttribute("href", "https://github.com/BerriAI/litellm/discussions/32090");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("exposes a named close control", () => {
    render(<DeprecationBanner featureName="Memory" />);

    expect(screen.getByRole("button", { name: /close/i })).toBeInTheDocument();
  });

  it("hides the banner once the close control is used", async () => {
    const user = userEvent.setup();
    render(<DeprecationBanner featureName="Memory" />);

    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(screen.queryByText("Memory is on a draft deprecation list")).not.toBeInTheDocument();
  });
});
