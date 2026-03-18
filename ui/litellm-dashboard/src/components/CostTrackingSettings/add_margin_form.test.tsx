import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import AddMarginForm from "./add_margin_form";
import { MarginConfig } from "./types";

vi.mock("../provider_info_helpers", () => ({
  Providers: {
    OpenAI: "OpenAI",
    Anthropic: "Anthropic",
  },
  provider_map: {
    OpenAI: "openai",
    Anthropic: "anthropic",
  },
  providerLogoMap: {
    OpenAI: "https://example.com/openai.png",
    Anthropic: "https://example.com/anthropic.png",
  },
}));

vi.mock("./provider_display_helpers", () => ({
  handleImageError: vi.fn(),
}));

const DEFAULT_PROPS = {
  marginConfig: {} as MarginConfig,
  selectedProvider: undefined,
  marginType: "percentage" as const,
  percentageValue: "",
  fixedAmountValue: "",
  onProviderChange: vi.fn(),
  onMarginTypeChange: vi.fn(),
  onPercentageChange: vi.fn(),
  onFixedAmountChange: vi.fn(),
  onAddProvider: vi.fn(),
};

describe("AddMarginForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} />);
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeInTheDocument();
  });

  it("should show the percentage input when marginType is percentage", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} marginType="percentage" />);
    expect(screen.getByPlaceholderText("10")).toBeInTheDocument();
  });

  it("should show the fixed amount input when marginType is fixed", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} marginType="fixed" />);
    expect(screen.getByPlaceholderText("0.001")).toBeInTheDocument();
  });

  it("should not show the fixed amount input when marginType is percentage", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} marginType="percentage" />);
    expect(screen.queryByPlaceholderText("0.001")).not.toBeInTheDocument();
  });

  it("should not show the percentage input when marginType is fixed", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} marginType="fixed" />);
    expect(screen.queryByPlaceholderText("10")).not.toBeInTheDocument();
  });

  it("should show the Percentage-based and Fixed Amount radio options", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} />);
    expect(screen.getByText("Percentage-based")).toBeInTheDocument();
    expect(screen.getByText("Fixed Amount")).toBeInTheDocument();
  });

  it("should disable the submit button when no provider is selected (percentage mode)", () => {
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider={undefined} percentageValue="10" />
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should disable the submit button when provider is selected but no percentage value (percentage mode)", () => {
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" percentageValue="" />
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should enable the submit button when provider and percentage value are both provided", () => {
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" percentageValue="10" />
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).not.toBeDisabled();
  });

  it("should disable the submit button in fixed mode when no fixed amount is provided", () => {
    renderWithProviders(
      <AddMarginForm
        {...DEFAULT_PROPS}
        selectedProvider="OpenAI"
        marginType="fixed"
        fixedAmountValue=""
      />
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should enable the submit button in fixed mode when provider and fixed amount are provided", () => {
    renderWithProviders(
      <AddMarginForm
        {...DEFAULT_PROPS}
        selectedProvider="OpenAI"
        marginType="fixed"
        fixedAmountValue="0.001"
      />
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).not.toBeDisabled();
  });

  it("should call onAddProvider when the enabled submit button is clicked", async () => {
    const onAddProvider = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddMarginForm
        {...DEFAULT_PROPS}
        selectedProvider="OpenAI"
        percentageValue="10"
        onAddProvider={onAddProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /add provider margin/i }));
    expect(onAddProvider).toHaveBeenCalledTimes(1);
  });

  it("should call onMarginTypeChange when the Fixed Amount radio is clicked", async () => {
    const onMarginTypeChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} onMarginTypeChange={onMarginTypeChange} />
    );

    await user.click(screen.getByText("Fixed Amount"));
    expect(onMarginTypeChange).toHaveBeenCalledWith("fixed");
  });
});
