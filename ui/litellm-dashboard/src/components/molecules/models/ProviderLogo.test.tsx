import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, render, fireEvent, act } from "@testing-library/react";
import { ProviderLogo } from "./ProviderLogo";
import * as providerInfoHelpers from "../../provider_info_helpers";

vi.mock("../../provider_info_helpers");

describe("ProviderLogo", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("should render the component", () => {
    vi.mocked(providerInfoHelpers.getProviderLogoAndName).mockReturnValue({
      logo: "",
      displayName: "Test Provider",
    });
    render(<ProviderLogo provider="test" />);
    expect(screen.getByText("t")).toBeInTheDocument();
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
