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
    expect(img).toHaveAttribute("src", providerLogoMap[Providers.OpenAI]);
    expect(img).toHaveAttribute("src", expect.stringContaining("openai_small"));
  });

  it("renders a letter avatar and no img for an unknown provider", () => {
    render(<Logo provider="unknown_provider_xyz" />);
    expect(screen.getByText("u")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("renders a dash avatar when src is empty and the label has no characters", () => {
    render(<Logo src={null} label="" />);
    expect(screen.getByText("-")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("resolves a backend asset path through the server root path in src mode", () => {
    render(<Logo src="/ui/assets/logos/github.svg" label="GitHub" />);
    const img = screen.getByRole("img", { name: "GitHub logo" });
    expect(img).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");
  });

  it("passes an external https URL through untouched in src mode", () => {
    render(<Logo src="https://cdn.example.com/logo.png" label="Ext" />);
    expect(screen.getByRole("img")).toHaveAttribute("src", "https://cdn.example.com/logo.png");
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

  it("leaves the caller's class list untouched for an asset that reads on dark", () => {
    render(<Logo src="/ui/assets/logos/slack.svg" label="Slack" className="w-5 h-5 shrink-0" />);
    expect(screen.getByRole("img", { name: "Slack logo" })).toHaveClass("w-5 h-5 shrink-0", { exact: true });
  });

  it("passes an untreated logo's classes through verbatim rather than normalizing them", () => {
    render(<Logo src="/ui/assets/logos/slack.svg" label="Slack" className="w-4 w-5 h-5" />);
    expect(screen.getByRole("img", { name: "Slack logo" })).toHaveClass("w-4 w-5 h-5", { exact: true });
  });

  it("forces a monochrome mark to white on dark without disturbing the caller's classes", () => {
    render(<Logo src="/ui/assets/logos/github.svg" label="GitHub" className="w-5 h-5" />);
    const img = screen.getByRole("img", { name: "GitHub logo" });
    expect(img).toHaveClass("w-5", "h-5", "dark:[filter:brightness(0)_invert(1)]");
    expect(img).not.toHaveClass("dark:bg-logo-surface");
  });

  it("plates a multicolor dark mark rather than inverting it", () => {
    render(<Logo src="/ui/assets/logos/fireworks.svg" label="Fireworks" className="w-5 h-5" />);
    const img = screen.getByRole("img", { name: "Fireworks logo" });
    expect(img).toHaveClass("dark:bg-logo-surface", "dark:object-contain", "dark:p-0.5");
    expect(img).not.toHaveClass("dark:[filter:brightness(0)_invert(1)]");
  });

  it("does not treat an external logo URL that collides with a bundled filename", () => {
    render(<Logo src="https://cdn.example.com/github.svg" label="Ext" className="w-5 h-5" />);
    expect(screen.getByRole("img", { name: "Ext logo" })).toHaveClass("w-5 h-5", { exact: true });
  });

  it("applies the treatment to a provider logo resolved through the bundler", () => {
    render(<Logo provider="openrouter" className="w-5 h-5" />);
    expect(screen.getByRole("img", { name: "openrouter logo" })).toHaveClass("dark:[filter:brightness(0)_invert(1)]");
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
    expect(img).toHaveAttribute("src", "/litellm/ui/assets/logos/github.svg");

    rerender(<Logo src="/ui/assets/logos/broken.svg" label="Agent" />);
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
    warnSpy.mockRestore();
  });
});
