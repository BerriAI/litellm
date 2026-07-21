import React from "react";
import { describe, expect, it, vi } from "vitest";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { Logo } from "./Logo";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";

vi.mock("@/lib/serverRootPath", () => ({ serverRootPath: "/litellm" }));

describe("Logo", () => {
  it("renders the bundled logo untouched by the server root path for a known provider", () => {
    render(<Logo provider="openai" />);
    const img = screen.getByRole("img", { name: "openai logo" });
    expect(img.getAttribute("src")).toBe(providerLogoMap[Providers.OpenAI]);
    expect(img.getAttribute("src")).toContain("openai_small");
  });

  it("renders a letter avatar and no img for an unknown provider", () => {
    render(<Logo provider="unknown_provider_xyz" />);
    expect(screen.getByText("u")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("renders a dash avatar when neither provider nor src is given", () => {
    render(<Logo />);
    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("resolves a backend asset path through the server root path in src mode", () => {
    render(<Logo src="/ui/assets/logos/github.svg" label="GitHub" />);
    const img = screen.getByRole("img", { name: "GitHub logo" });
    expect(img.getAttribute("src")).toBe("/litellm/ui/assets/logos/github.svg");
  });

  it("passes an external https URL through untouched in src mode", () => {
    render(<Logo src="https://cdn.example.com/logo.png" label="Ext" />);
    expect(screen.getByRole("img").getAttribute("src")).toBe("https://cdn.example.com/logo.png");
  });

  it("swaps to the letter avatar and warns with the failing URL on image error", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    render(<Logo src="/ui/assets/logos/github.svg" label="GitHub" />);
    const img = screen.getByRole("img", { name: "GitHub logo" });

    act(() => {
      fireEvent.error(img);
    });

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("G")).toBeInTheDocument();
    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("/litellm/ui/assets/logos/github.svg"));
    warnSpy.mockRestore();
  });

  it("retries with a new src after a previous src errored", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { rerender } = render(<Logo src="/ui/assets/logos/broken.svg" label="Agent" />);

    act(() => {
      fireEvent.error(screen.getByRole("img", { name: "Agent logo" }));
    });
    expect(screen.queryByRole("img")).not.toBeInTheDocument();

    rerender(<Logo src="/ui/assets/logos/github.svg" label="Agent" />);
    const img = screen.getByRole("img", { name: "Agent logo" });
    expect(img.getAttribute("src")).toBe("/litellm/ui/assets/logos/github.svg");

    rerender(<Logo src="/ui/assets/logos/broken.svg" label="Agent" />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
    warnSpy.mockRestore();
  });
});
