import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import ProviderDiscountTable from "./provider_discount_table";

const DEFAULT_DISCOUNT_CONFIG = {
  openai: 0.05,
  anthropic: 0.1,
};

const ROW_ACTION_NAME = {
  edit: /^Edit discount for /,
  save: /^Save discount for /,
  cancel: /^Cancel editing discount for /,
  remove: /^Remove discount for /,
} as const;

const rowAction = (action: keyof typeof ROW_ACTION_NAME): HTMLElement =>
  screen.getByRole("button", { name: ROW_ACTION_NAME[action] });

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
    expect(screen.getByRole("columnheader", { name: "Provider" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Discount Percentage" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Actions" })).toBeInTheDocument();
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

  it("should sort rows by provider display name", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={DEFAULT_DISCOUNT_CONFIG}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining("Anthropic"),
      expect.stringContaining("OpenAI"),
    ]);
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

  it("should render the provider logo alongside the display name", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByRole("img", { name: "OpenAI logo" })).toBeInTheDocument();
  });

  it("should show a text input when the edit icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

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

    await user.click(rowAction("edit"));

    expect(screen.queryByText("5.0%")).not.toBeInTheDocument();
  });

  it("should seed the edit input with the current discount as a percentage", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    expect(screen.getByPlaceholderText("5")).toHaveValue("5");
  });

  it("should call onDiscountChange with the new value when the save icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    fireEvent.change(input, { target: { value: "10" } });

    await user.click(rowAction("save"));

    expect(onDiscountChange).toHaveBeenCalledWith("openai", "0.1");
  });

  it("should save the edited discount when Enter is pressed", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "10{Enter}");

    expect(onDiscountChange).toHaveBeenCalledWith("openai", "0.1");
    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
  });

  it("should abandon the edit when Escape is pressed", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "10{Escape}");

    expect(onDiscountChange).not.toHaveBeenCalled();
    expect(screen.getByText("5.0%")).toBeInTheDocument();
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

    await user.click(rowAction("edit"));
    await user.click(rowAction("save"));

    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
  });

  it("should cancel edit mode when the cancel icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));
    await user.click(rowAction("cancel"));

    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
    expect(onDiscountChange).not.toHaveBeenCalled();
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  it("should call onRemoveProvider with the provider key and display name when the trash icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("remove"));

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

    await user.click(rowAction("edit"));
    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    fireEvent.change(input, { target: { value: "150" } });
    await user.click(rowAction("save"));

    expect(onDiscountChange).not.toHaveBeenCalled();
  });

  it("should expose each row action as a button named for its provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    expect(screen.getByRole("button", { name: "Edit discount for OpenAI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove discount for OpenAI" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit discount for OpenAI" }));

    expect(screen.getByRole("button", { name: "Save discount for OpenAI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel editing discount for OpenAI" })).toBeInTheDocument();
  });

  it("should render the empty message when no discounts are configured", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{}}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("No provider discounts configured")).toBeInTheDocument();
  });
});
