import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import MoyaiLanding, { MOYAI_GITHUB_URL, MOYAI_LAUNCH_POST_URL, MOYAI_WALKTHROUGH_URL } from "./MoyaiLanding";

vi.mock("./moyaiSky", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./moyaiSky")>();
  return {
    ...actual,
    prefersReducedMotion: () => true,
    startStarfield: () => () => {},
    startPlanetrise: () => () => {},
  };
});

describe("MoyaiLanding", () => {
  it("shows the admin quick-connect line and validates the dialog input", async () => {
    const onQuickConnect = vi.fn(async () => {});
    render(<MoyaiLanding canQuickConnect onQuickConnect={onQuickConnect} />);

    fireEvent.click(screen.getByRole("button", { name: "Quick connect" }));

    const input = await screen.findByLabelText("Moyai URL");
    fireEvent.change(input, { target: { value: "javascript:alert(1)" } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("http or https");
    expect(onQuickConnect).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: "  https://moyai.example.com/  " } });
    fireEvent.click(screen.getByRole("button", { name: "Connect" }));
    expect(onQuickConnect).toHaveBeenCalledWith("https://moyai.example.com");
  });

  it("shows the non-admin copy when quick connect is unavailable", () => {
    render(<MoyaiLanding />);

    expect(screen.getByText(/Ask a proxy admin to connect it/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Quick connect" })).not.toBeInTheDocument();
  });

  it("links the GitHub, demo, and launch post CTAs to the exported URLs", () => {
    render(<MoyaiLanding />);

    expect(screen.getByRole("link", { name: /Star Moyai on GitHub/ })).toHaveAttribute("href", MOYAI_GITHUB_URL);
    expect(screen.getByRole("link", { name: /Watch the demo/ })).toHaveAttribute("href", MOYAI_WALKTHROUGH_URL);
    expect(screen.getByRole("link", { name: /Read the launch post/ })).toHaveAttribute("href", MOYAI_LAUNCH_POST_URL);
  });

  it("shows the GitHub fallback when the demo image fails to load", () => {
    render(<MoyaiLanding />);

    const demoImg = screen.getByAltText(/Moyai demo:/);
    fireEvent.error(demoImg);

    expect(screen.getByText("Watch the demo on GitHub")).toBeInTheDocument();
  });
});
