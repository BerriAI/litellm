import type { ComponentProps } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import ModelHubProviders, { ProviderLogo } from "./ModelHubProviders";

vi.mock("next/image", () => ({
  default: ({ unoptimized: _unoptimized, alt, ...props }: ComponentProps<"img"> & { unoptimized?: boolean }) => (
    // eslint-disable-next-line @next/next/no-img-element -- Native test double for Next's image component.
    <img alt={alt} {...props} />
  ),
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({ variant: _variant, ...props }: ComponentProps<"button"> & { variant?: string }) => <button {...props} />,
}));

vi.mock("@/components/ui/skeleton", () => ({
  Skeleton: (props: ComponentProps<"div">) => <div {...props} />,
}));

vi.mock("@/components/provider_info_helpers", () => ({
  getProviderLogoAndName: (provider: string) => {
    if (provider === "openai") return { logo: "/openai.svg", displayName: "OpenAI" };
    if (provider === "anthropic") return { logo: "/anthropic.svg", displayName: "Anthropic" };
    return { logo: "", displayName: provider };
  },
}));

describe("ModelHubProviders", () => {
  it("should render available providers with clear accessible names and the current selection", () => {
    render(
      <ModelHubProviders providers={["openai", "anthropic"]} selectedProviders={["anthropic"]} onSelect={vi.fn()} />,
    );

    expect(screen.getByRole("group", { name: "Available providers" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "OpenAI", pressed: false })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Anthropic", pressed: true })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "All models", pressed: false })).toBeInTheDocument();
  });

  it("should select a provider and clear the selection with All models", () => {
    const onSelect = vi.fn();
    render(<ModelHubProviders providers={["openai"]} selectedProviders={[]} onSelect={onSelect} />);

    expect(screen.getByRole("button", { name: "All models", pressed: true })).toBeInTheDocument();
    act(() => fireEvent.click(screen.getByRole("button", { name: "OpenAI" })));
    expect(onSelect).toHaveBeenLastCalledWith("openai");
    act(() => fireEvent.click(screen.getByRole("button", { name: "All models" })));
    expect(onSelect).toHaveBeenLastCalledWith(null);
  });

  it("should announce provider loading before the available providers arrive", () => {
    render(<ModelHubProviders providers={[]} selectedProviders={[]} onSelect={vi.fn()} loading />);

    expect(screen.getByRole("group", { name: "Available providers" })).toHaveAttribute("aria-busy", "true");
    expect(screen.getByRole("status")).toHaveTextContent("Loading providers");
  });
});

describe("ProviderLogo", () => {
  it("should show a provider initial when its logo is unknown", () => {
    render(<ProviderLogo provider="local-models" />);

    expect(screen.getByText("L")).toBeInTheDocument();
    expect(screen.queryByAltText("")).not.toBeInTheDocument();
  });

  it("should fall back after a broken logo and load a different provider's logo", () => {
    const { rerender } = render(<ProviderLogo provider="openai" />);

    act(() => fireEvent.error(screen.getByAltText("")));
    expect(screen.getByText("O")).toBeInTheDocument();
    expect(screen.queryByAltText("")).not.toBeInTheDocument();

    rerender(<ProviderLogo provider="anthropic" />);
    expect(screen.getByAltText("")).toHaveAttribute("src", "/anthropic.svg");
    expect(screen.queryByText("O")).not.toBeInTheDocument();
  });
});
