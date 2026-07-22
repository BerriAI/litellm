import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import AddProviderForm from "./add_provider_form";
import { DiscountConfig } from "./types";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";

const DEFAULT_PROPS = {
  discountConfig: {} as DiscountConfig,
  selectedProvider: undefined,
  newDiscount: "",
  onProviderChange: vi.fn(),
  onDiscountChange: vi.fn(),
  onAddProvider: vi.fn(),
};

describe("AddProviderForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} />);
    expect(screen.getByRole("button", { name: /add provider discount/i })).toBeInTheDocument();
  });

  it("should render the discount percentage input field", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} />);
    expect(screen.getByPlaceholderText("5")).toBeInTheDocument();
  });

  it("should disable the submit button when no provider is selected and no discount is entered", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} />);
    expect(screen.getByRole("button", { name: /add provider discount/i })).toBeDisabled();
  });

  it("should disable the submit button when a provider is selected but no discount is entered", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} selectedProvider="OpenAI" newDiscount="" />);
    expect(screen.getByRole("button", { name: /add provider discount/i })).toBeDisabled();
  });

  it("should disable the submit button when a discount is entered but no provider is selected", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} selectedProvider={undefined} newDiscount="5" />);
    expect(screen.getByRole("button", { name: /add provider discount/i })).toBeDisabled();
  });

  it("should enable the submit button when both a provider and a discount value are provided", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} selectedProvider="OpenAI" newDiscount="5" />);
    expect(screen.getByRole("button", { name: /add provider discount/i })).not.toBeDisabled();
  });

  it("should call onAddProvider when the enabled submit button is clicked", async () => {
    const onAddProvider = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddProviderForm {...DEFAULT_PROPS} selectedProvider="OpenAI" newDiscount="5" onAddProvider={onAddProvider} />,
    );

    await user.click(screen.getByRole("button", { name: /add provider discount/i }));
    expect(onAddProvider).toHaveBeenCalledTimes(1);
  });

  it("should show the percent sign next to the discount input", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} />);
    expect(screen.getByText("%")).toBeInTheDocument();
  });

  it("renders the selected provider's bundled logo via the shared Logo component", async () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} selectedProvider="OpenAI" />);

    const logo = await screen.findByRole("img", { name: `${Providers.OpenAI} logo` });
    expect(logo.getAttribute("src")).toBe(providerLogoMap[Providers.OpenAI]);
  });

  it("falls back to a letter avatar for a selected provider that has no bundled logo", () => {
    renderWithProviders(<AddProviderForm {...DEFAULT_PROPS} selectedProvider="PG_VECTOR" />);

    expect(screen.queryByRole("img", { name: `${Providers.PG_VECTOR} logo` })).not.toBeInTheDocument();
    expect(screen.getByText(Providers.PG_VECTOR.charAt(0))).toBeInTheDocument();
  });
});
