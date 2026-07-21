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

  it("should show a preview when a value is provided", () => {
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" />);
    expect(screen.getByAltText("Selected logo")).toBeInTheDocument();
  });

  it("should not show a preview when no value is provided", () => {
    render(<MCPLogoSelector />);
    expect(screen.queryByAltText("Selected logo")).not.toBeInTheDocument();
  });

  it("should call onChange with undefined when the clear button is clicked", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" onChange={onChange} />);

    await user.click(screen.getByRole("button", { name: /✕/ }));
    expect(onChange).toHaveBeenCalledWith(undefined);
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

  it("should render grid logos from bundled static assets instead of public paths", () => {
    render(<MCPLogoSelector />);
    const src = screen.getByAltText("GitHub").getAttribute("src");
    expect(src).toMatch(/^\/_next\//);
    expect(src).toContain("github.svg");
  });

  it("should preview a stored well-known path via its bundled asset", () => {
    render(<MCPLogoSelector value="/ui/assets/logos/github.svg" />);
    const src = screen.getByAltText("Selected logo").getAttribute("src");
    expect(src).toMatch(/^\/_next\//);
    expect(src).toContain("github.svg");
  });

  it("should preview a custom external URL untouched", () => {
    render(<MCPLogoSelector value="https://cdn.example.com/logo.png" />);
    expect(screen.getByAltText("Selected logo").getAttribute("src")).toBe("https://cdn.example.com/logo.png");
  });
});
