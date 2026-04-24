import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import ProviderDiscountTable from "./provider_discount_table";

vi.mock("./provider_display_helpers", () => ({
  getProviderDisplayInfo: vi.fn((providerValue: string) => ({
    displayName: providerValue === "openai" ? "OpenAI" : providerValue,
    logo: providerValue === "openai" ? "https://example.com/openai.png" : "",
    enumKey: providerValue === "openai" ? "OpenAI" : null,
  })),
  handleImageError: vi.fn(),
}));

const DEFAULT_DISCOUNT_CONFIG = {
  openai: 0.05,
  anthropic: 0.1,
};

describe("ProviderDiscountTable", () => {
  const onDiscountChange = vi.fn();
  const onRemoveProvider = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={DEFAULT_DISCOUNT_CONFIG}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should render the table headers", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={DEFAULT_DISCOUNT_CONFIG}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("Discount Percentage")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should display provider display names in the table", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={DEFAULT_DISCOUNT_CONFIG}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
  });

  it("should display the formatted discount percentage", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  it("should show a text input when the edit button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));

    expect(screen.getByPlaceholderText("5")).toBeInTheDocument();
  });

  it("should hide the formatted percentage when in edit mode", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));

    expect(screen.queryByText("5.0%")).not.toBeInTheDocument();
  });

  it("should call onDiscountChange with the new value when the save button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));

    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "10");

    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onDiscountChange).toHaveBeenCalledWith("openai", "0.1");
  });

  it("should restore the display view after saving", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
  });

  it("should cancel edit mode when the cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
    expect(onDiscountChange).not.toHaveBeenCalled();
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  it("should not call onDiscountChange when canceling edit", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(onDiscountChange).not.toHaveBeenCalled();
  });

  it("should call onRemoveProvider with the provider key and display name when Remove is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Remove" }));

    expect(onRemoveProvider).toHaveBeenCalledWith("openai", "OpenAI");
  });

  it("should not call onDiscountChange when the entered value is out of range", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Edit" }));
    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "150");
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(onDiscountChange).not.toHaveBeenCalled();
  });
});
