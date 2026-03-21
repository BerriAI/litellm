import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import ProviderDiscountTable from "./provider_discount_table";

vi.mock("@heroicons/react/outline", () => ({
  TrashIcon: function TrashIcon() { return null; },
  PencilAltIcon: function PencilAltIcon() { return null; },
  CheckIcon: function CheckIcon() { return null; },
  XIcon: function XIcon() { return null; },
}));

vi.mock("@tremor/react", () => ({
  Table: ({ children }: any) => <table>{children}</table>,
  TableHead: ({ children }: any) => <thead>{children}</thead>,
  TableRow: ({ children }: any) => <tr>{children}</tr>,
  TableHeaderCell: ({ children }: any) => <th>{children}</th>,
  TableBody: ({ children }: any) => <tbody>{children}</tbody>,
  TableCell: ({ children }: any) => <td>{children}</td>,
  Text: ({ children }: any) => <span>{children}</span>,
  TextInput: ({ value, onValueChange, onKeyDown, placeholder, ...rest }: any) => (
    <input
      value={value}
      onChange={(e) => onValueChange?.(e.target.value)}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      {...rest}
    />
  ),
  Icon: ({ icon: IconComponent, onClick }: any) => {
    const name = IconComponent?.displayName ?? IconComponent?.name ?? "icon";
    return <button onClick={onClick} aria-label={name} />;
  },
}));

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
      />
    );
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should render the table headers", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={DEFAULT_DISCOUNT_CONFIG}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
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
      />
    );
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
  });

  it("should display the formatted discount percentage", () => {
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("5.0%")).toBeInTheDocument();
  });

  it("should show a text input when the edit icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    const pencilButton = screen.getByRole("button", { name: /PencilAltIcon/i });
    await user.click(pencilButton);

    expect(screen.getByPlaceholderText("5")).toBeInTheDocument();
  });

  it("should hide the formatted percentage when in edit mode", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));

    expect(screen.queryByText("5.0%")).not.toBeInTheDocument();
  });

  it("should call onDiscountChange with the new value when the save icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));

    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "10");

    await user.click(screen.getByRole("button", { name: /CheckIcon/i }));

    expect(onDiscountChange).toHaveBeenCalledWith("openai", "0.1");
  });

  it("should restore the display view after saving", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));
    await user.click(screen.getByRole("button", { name: /CheckIcon/i }));

    expect(screen.queryByPlaceholderText("5")).not.toBeInTheDocument();
  });

  it("should cancel edit mode when the cancel icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));
    await user.click(screen.getByRole("button", { name: /XIcon/i }));

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
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));
    await user.click(screen.getByRole("button", { name: /XIcon/i }));

    expect(onDiscountChange).not.toHaveBeenCalled();
  });

  it("should call onRemoveProvider with the provider key and display name when the trash icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));

    expect(onRemoveProvider).toHaveBeenCalledWith("openai", "OpenAI");
  });

  it("should not call onDiscountChange when the entered value is out of range", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderDiscountTable
        discountConfig={{ openai: 0.05 }}
        onDiscountChange={onDiscountChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));
    const input = screen.getByPlaceholderText("5");
    await user.clear(input);
    await user.type(input, "150");
    await user.click(screen.getByRole("button", { name: /CheckIcon/i }));

    expect(onDiscountChange).not.toHaveBeenCalled();
  });
});
