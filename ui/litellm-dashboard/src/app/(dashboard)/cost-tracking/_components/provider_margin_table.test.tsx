import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import ProviderMarginTable from "./provider_margin_table";
import { Providers, providerLogoMap } from "@/components/provider_info_helpers";

const ROW_ACTION_NAME = {
  edit: /^Edit margin for /,
  save: /^Save margin for /,
  cancel: /^Cancel editing margin for /,
  remove: /^Remove margin for /,
} as const;

const rowAction = (action: keyof typeof ROW_ACTION_NAME): HTMLElement =>
  screen.getByRole("button", { name: ROW_ACTION_NAME[action] });

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
      />,
    );
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should render the table headers", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByRole("columnheader", { name: "Provider" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Margin" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Actions" })).toBeInTheDocument();
  });

  it("should display the provider display name", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("OpenAI")).toBeInTheDocument();
  });

  it("should render the provider's bundled logo via the shared Logo component", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    const logo = screen.getByRole("img", { name: `${Providers.OpenAI} logo` });
    expect(logo).toHaveAttribute("src", providerLogoMap[Providers.OpenAI]);
  });

  it("should fall back to a letter avatar for a provider with no bundled logo", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ "my-custom-provider": 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    expect(screen.getByText("m")).toBeInTheDocument();
  });

  it("should display the global provider as 'Global (All Providers)'", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("Global (All Providers)")).toBeInTheDocument();
  });

  it("should sort the global row above provider rows", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1, global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    const rows = screen.getAllByRole("row").slice(1);
    expect(rows.map((row) => row.textContent)).toEqual([
      expect.stringContaining("Global (All Providers)"),
      expect.stringContaining("OpenAI"),
    ]);
  });

  it("should display a numeric margin as a percentage", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("10.0%")).toBeInTheDocument();
  });

  it("should display a fixed amount margin with dollar sign", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: { fixed_amount: 0.001 } }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );
    expect(screen.getByText("$0.001000")).toBeInTheDocument();
  });

  it("should display a combined percentage and fixed margin", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: { percentage: 0.1, fixed_amount: 0.001 } }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
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
      />,
    );

    await user.click(rowAction("edit"));

    expect(screen.getByPlaceholderText("10")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("0.001")).toBeInTheDocument();
  });

  it("should seed the percentage input from a numeric margin and leave the fixed amount blank", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    expect(screen.getByPlaceholderText("10")).toHaveValue("10");
    expect(screen.getByPlaceholderText("0.001")).toHaveValue("");
  });

  it("should call onMarginChange with a percentage value when save is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    const percentInput = screen.getByPlaceholderText("10");
    await user.clear(percentInput);
    fireEvent.change(percentInput, { target: { value: "20" } });

    await user.click(rowAction("save"));

    expect(onMarginChange).toHaveBeenCalledWith("openai", 0.2);
  });

  it("should call onMarginChange with a fixed-amount-only object when the percentage is cleared", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));

    await user.clear(screen.getByPlaceholderText("10"));
    fireEvent.change(screen.getByPlaceholderText("0.001"), { target: { value: "0.002" } });

    await user.click(rowAction("save"));

    expect(onMarginChange).toHaveBeenCalledWith("openai", { fixed_amount: 0.002 });
  });

  it("should cancel edit mode without calling onMarginChange when X is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("edit"));
    await user.click(rowAction("cancel"));

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
      />,
    );

    await user.click(rowAction("remove"));

    expect(onRemoveProvider).toHaveBeenCalledWith("openai", "OpenAI");
  });

  it("should call onRemoveProvider with 'Global' display name for the global provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    await user.click(rowAction("remove"));

    expect(onRemoveProvider).toHaveBeenCalledWith("global", "Global");
  });

  it("should expose each row action as a button named for its provider", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ openai: 0.1 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    expect(screen.getByRole("button", { name: "Edit margin for OpenAI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove margin for OpenAI" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Edit margin for OpenAI" }));

    expect(screen.getByRole("button", { name: "Save margin for OpenAI" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cancel editing margin for OpenAI" })).toBeInTheDocument();
  });

  it("should name the global row's actions after the global provider", () => {
    renderWithProviders(
      <ProviderMarginTable
        marginConfig={{ global: 0.05 }}
        onMarginChange={onMarginChange}
        onRemoveProvider={onRemoveProvider}
      />,
    );

    expect(screen.getByRole("button", { name: "Edit margin for Global" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove margin for Global" })).toBeInTheDocument();
  });

  it("should render the empty message when no margins are configured", () => {
    renderWithProviders(
      <ProviderMarginTable marginConfig={{}} onMarginChange={onMarginChange} onRemoveProvider={onRemoveProvider} />,
    );
    expect(screen.getByText("No provider margins configured")).toBeInTheDocument();
  });

  describe("when both percentage and fixed amount are entered", () => {
    it("should call onMarginChange with an object containing both values", async () => {
      const user = userEvent.setup();
      renderWithProviders(
        <ProviderMarginTable
          marginConfig={{ openai: 0.1 }}
          onMarginChange={onMarginChange}
          onRemoveProvider={onRemoveProvider}
        />,
      );

      await user.click(rowAction("edit"));

      const percentInput = screen.getByPlaceholderText("10");
      await user.clear(percentInput);
      fireEvent.change(percentInput, { target: { value: "5" } });

      const fixedInput = screen.getByPlaceholderText("0.001");
      fireEvent.change(fixedInput, { target: { value: "0.002" } });

      await user.click(rowAction("save"));

      expect(onMarginChange).toHaveBeenCalledWith("openai", {
        percentage: 0.05,
        fixed_amount: 0.002,
      });
    });
  });
});
