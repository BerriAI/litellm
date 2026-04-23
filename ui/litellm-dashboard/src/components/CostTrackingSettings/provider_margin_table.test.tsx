import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import ProviderMarginTable from "./provider_margin_table";

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
  TextInput: ({ value, onValueChange, placeholder, autoFocus, className }: any) => (
    <input
      value={value}
      onChange={(e) => onValueChange?.(e.target.value)}
      placeholder={placeholder}
      autoFocus={autoFocus}
      className={className}
    />
  ),
  Icon: ({ icon: IconComponent, onClick }: any) => {
    const name = IconComponent?.displayName ?? IconComponent?.name ?? "icon";
    return <button onClick={onClick} aria-label={name} />;
  },
}));

vi.mock("./provider_display_helpers", () => ({
  getProviderDisplayInfo: vi.fn((providerValue: string) => {
    if (providerValue === "openai") return { displayName: "OpenAI", logo: "", enumKey: "OpenAI" };
    if (providerValue === "anthropic") return { displayName: "Anthropic", logo: "", enumKey: "Anthropic" };
    return { displayName: providerValue, logo: "", enumKey: null };
  }),
  handleImageError: vi.fn(),
}));

describe("ProviderMarginTable", () => {
  const onMarginChange = vi.fn();
  const onRemoveProvider = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should render the table headers", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("Margin")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should display the provider display name", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
  });

  it("should display the global provider as 'Global (All Providers)'", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("Global (All Providers)")).toBeInTheDocument();
  });

  it("should display a numeric margin as a percentage", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("10.0%")).toBeInTheDocument();
  });

  it("should display a fixed amount margin with dollar sign", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: { fixed_amount: 0.001 } }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText("$0.001000")).toBeInTheDocument();
  });

  it("should display a combined percentage and fixed margin", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: { percentage: 0.1, fixed_amount: 0.001 } }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );
    expect(screen.getByText(/10\.0%.*\$0\.001000/)).toBeInTheDocument();
  });

  it("should show edit inputs when the pencil icon is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));

    expect(screen.getByPlaceholderText("10")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("0.001")).toBeInTheDocument();
  });

  it("should call onMarginChange with a percentage value when save is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));

    const percentInput = screen.getByPlaceholderText("10");
    await user.clear(percentInput);
    await user.type(percentInput, "20");

    await user.click(screen.getByRole("button", { name: /CheckIcon/i }));

    expect(onMarginChange).toHaveBeenCalledWith("openai", 0.2);
  });

  it("should cancel edit mode without calling onMarginChange when X is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));
    await user.click(screen.getByRole("button", { name: /XIcon/i }));

    expect(onMarginChange).not.toHaveBeenCalled();
    expect(screen.queryByPlaceholderText("10")).not.toBeInTheDocument();
  });

  it("should call onRemoveProvider with provider key and display name when trash is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));

    expect(onRemoveProvider).toHaveBeenCalledWith("openai", "OpenAI");
  });

  it("should call onRemoveProvider with 'Global' display name for the global provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />
    );

    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));

    expect(onRemoveProvider).toHaveBeenCalledWith("global", "Global");
  });

  describe("when both percentage and fixed amount are entered", () => {
    it("should call onMarginChange with an object containing both values", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ProviderMarginTable
          marginConfig={{ openai: 0.1 }}
          onMarginChange={onMarginChange}
          onRemoveProvider={onRemoveProvider}
        />
      );

      await user.click(screen.getByRole("button", { name: /PencilAltIcon/i }));

      const percentInput = screen.getByPlaceholderText("10");
      await user.clear(percentInput);
      await user.type(percentInput, "5");

      const fixedInput = screen.getByPlaceholderText("0.001");
      await user.type(fixedInput, "0.002");

      await user.click(screen.getByRole("button", { name: /CheckIcon/i }));

      expect(onMarginChange).toHaveBeenCalledWith("openai", {
        percentage: 0.05,
        fixed_amount: 0.002,
      });
    });
  });
});
