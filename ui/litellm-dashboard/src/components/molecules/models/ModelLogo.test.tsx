import React from "react";
import { describe, it, expect } from "vitest";
import { screen, render } from "@testing-library/react";
import { ModelLogo } from "./ModelLogo";
import { getProviderLogoAndName } from "../../provider_info_helpers";

describe("ModelLogo", () => {
  it("renders the model family logo instead of the aggregating provider logo", () => {
    render(<ModelLogo model="scaleway/qwen3-235b-a22b-instruct-2507" provider="openrouter" />);
    const img = screen.getByRole("img", { name: "openrouter logo" });
    expect(img.getAttribute("src")).toContain("qwen");
  });

  it("renders the provider logo when the model names no known family", () => {
    render(<ModelLogo model="openrouter/some-unknown-model" provider="openrouter" />);
    const img = screen.getByRole("img", { name: "openrouter logo" });
    expect(img.getAttribute("src")).toBe(getProviderLogoAndName("openrouter").logo);
  });
});
