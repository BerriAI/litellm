import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, render, fireEvent, act } from "@testing-library/react";
import { ProviderLogo } from "./ProviderLogo";
import * as providerInfoHelpers from "../../provider_info_helpers";

vi.mock("../../provider_info_helpers");
vi.mock("@/lib/serverRootPath", () => ({ serverRootPath: "/litellm" }));

describe("ProviderLogo", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    vi.mocked(providerInfoHelpers.getProviderLogoAndName).mockReturnValue({
      logo: "/assets/provider-logo.svg",
      displayName: "Test Provider",
    });
  });

  it("should render the component", () => {
    vi.mocked(providerInfoHelpers.getProviderLogoAndName).mockReturnValue({
      logo: "",
      displayName: "Test Provider",
    });
    render(<ProviderLogo provider="test" />);
    expect(screen.getByText("t")).toBeInTheDocument();
  });

  it.each([
    ["https://cdn.example.com/gateway.svg", "https://cdn.example.com/gateway.svg"],
    ["/ui/assets/logos/gateway.svg", "/litellm/ui/assets/logos/gateway.svg"],
  ])("should display a custom logo from %s", (logoUrl, expectedSrc) => {
    render(<ProviderLogo provider="test" logoUrl={logoUrl} />);

    expect(screen.getByRole("img", { name: "test logo" })).toHaveAttribute("src", expectedSrc);
  });

  it.each([undefined, null, ""])("should use the provider logo when the override is %s", (logoUrl) => {
    render(<ProviderLogo provider="test" logoUrl={logoUrl} />);

    expect(screen.getByRole("img", { name: "test logo" })).toHaveAttribute("src", "/assets/provider-logo.svg");
  });

  it("should recover when a failed custom logo is replaced", () => {
    const { rerender } = render(<ProviderLogo provider="test" logoUrl="https://cdn.example.com/broken.svg" />);

    act(() => {
      fireEvent.error(screen.getByRole("img", { name: "test logo" }));
    });

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("t")).toBeInTheDocument();

    rerender(<ProviderLogo provider="test" logoUrl="https://cdn.example.com/gateway.svg" />);

    expect(screen.getByRole("img", { name: "test logo" })).toHaveAttribute(
      "src",
      "https://cdn.example.com/gateway.svg",
    );
  });

  it("should show fallback when image fails to load", () => {
    vi.mocked(providerInfoHelpers.getProviderLogoAndName).mockReturnValue({
      logo: "/path/to/logo.png",
      displayName: "Test Provider",
    });
    render(<ProviderLogo provider="test" />);
    const img = screen.getByRole("img", { name: "test logo" });
    expect(img).toBeInTheDocument();

    act(() => {
      fireEvent.error(img);
    });

    expect(screen.getByText("t")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
