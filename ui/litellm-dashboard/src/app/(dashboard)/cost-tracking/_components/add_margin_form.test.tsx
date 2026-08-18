import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import AddMarginForm from "./add_margin_form";
import { MarginConfig } from "./types";

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
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} selectedProvider={undefined} percentageValue="10" />);
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should disable the submit button when provider is selected but no percentage value (percentage mode)", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" percentageValue="" />);
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should enable the submit button when provider and percentage value are both provided", () => {
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" percentageValue="10" />);
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeEnabled();
  });

  it("should disable the submit button in fixed mode when no fixed amount is provided", () => {
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" marginType="fixed" fixedAmountValue="" />,
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeDisabled();
  });

  it("should enable the submit button in fixed mode when provider and fixed amount are provided", () => {
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" marginType="fixed" fixedAmountValue="0.001" />,
    );
    expect(screen.getByRole("button", { name: /add provider margin/i })).toBeEnabled();
  });

  it("should call onAddProvider when the enabled submit button is clicked", async () => {
    const onAddProvider = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} selectedProvider="OpenAI" percentageValue="10" onAddProvider={onAddProvider} />,
    );

    await user.click(screen.getByRole("button", { name: /add provider margin/i }));
    expect(onAddProvider).toHaveBeenCalledTimes(1);
  });

  it("should report the edited percentage as the user types", async () => {
    const onPercentageChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddMarginForm {...DEFAULT_PROPS} percentageValue="1" onPercentageChange={onPercentageChange} />,
    );

    await user.type(screen.getByPlaceholderText("10"), "0");
    expect(onPercentageChange).toHaveBeenCalledWith("10");
  });

  it("should report the edited fixed amount as the user types", async () => {
    const onFixedAmountChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <AddMarginForm
        {...DEFAULT_PROPS}
        marginType="fixed"
        fixedAmountValue="0.00"
        onFixedAmountChange={onFixedAmountChange}
      />,
    );

    await user.type(screen.getByPlaceholderText("0.001"), "1");
    expect(onFixedAmountChange).toHaveBeenCalledWith("0.001");
  });

  it("should call onMarginTypeChange when the Fixed Amount radio is clicked", async () => {
    const onMarginTypeChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} onMarginTypeChange={onMarginTypeChange} />);

    await user.click(screen.getByText("Fixed Amount"));
    expect(onMarginTypeChange).toHaveBeenCalledWith("fixed");
  });

  it("should call onProviderChange with the provider key when a provider is picked", async () => {
    const onProviderChange = vi.fn();
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderWithProviders(<AddMarginForm {...DEFAULT_PROPS} onProviderChange={onProviderChange} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Anthropic"));

    expect(onProviderChange.mock.calls).toHaveLength(1);
    expect(onProviderChange.mock.calls[0]?.[0]).toBe("Anthropic");
  });
});
