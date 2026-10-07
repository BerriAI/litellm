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
