import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import MCPLogoSelector from "./MCPLogoSelector";

describe("MCPLogoSelector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the logo grid and custom URL input", () => {
    render(<MCPLogoSelector />);
    expect(screen.getByPlaceholderText(/paste a custom logo URL/i)).toBeInTheDocument();
  });

  it("should show the selected grid tile with a highlighted state", () => {
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" />);
    // Selection is conveyed by the blue border on the matching tile button,
    // not a separate preview banner. We check via the alt text on the inner
    // <img>, which equals the logo name.
    const githubImg = screen.getByAltText("GitHub");
    const tileButton = githubImg.closest("button");
    expect(tileButton).not.toBeNull();
    expect(tileButton?.className).toContain("border-blue-500");
  });

  it("should leave the custom URL input empty when a grid logo is selected", () => {
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" />);
    expect(screen.getByPlaceholderText(/paste a custom logo URL/i)).toHaveValue("");
  });

  it("should populate the custom URL input when value is not in the grid", () => {
    render(<MCPLogoSelector value="https://cdn.example.com/custom.png" />);
    expect(screen.getByPlaceholderText(/paste a custom logo URL/i)).toHaveValue(
      "https://cdn.example.com/custom.png",
    );
  });

  it("should call onChange with the logo URL when a grid logo is clicked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MCPLogoSelector onChange={onChange} />);

    const githubButton = screen.getByRole("button", { name: /GitHub/i });
    await user.click(githubButton);
    expect(onChange).toHaveBeenCalledWith("/ui/assets/logos/github.svg");
  });

  it("should deselect a logo when clicking the already-selected logo", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" onChange={onChange} />);

    const githubButton = screen.getByRole("button", { name: /GitHub/i });
    await user.click(githubButton);
    expect(onChange).toHaveBeenCalledWith(undefined);
  });
});
