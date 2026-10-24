import { useState } from "react";
import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { chooseSelectOption } from "@/../tests/test-utils";

import ModelHubProviders, { ProviderLogo } from "./ModelHubProviders";

function ProviderFilter({ initialProvider = "" }: { initialProvider?: string }) {
  const [provider, setProvider] = useState<string | null>(initialProvider);
  return (
    <ModelHubProviders
      providers={["openai", "anthropic"]}
      selectedProviders={provider ? [provider] : []}
      onSelect={setProvider}
    />
  );
}

describe("ModelHubProviders", () => {
  it("should render the selected provider and its decorative logo in the dropdown", () => {
    render(<ProviderFilter initialProvider="anthropic" />);

    const dropdown = screen.getByRole("combobox", { name: "Provider" });
    expect(dropdown).toHaveTextContent("Anthropic");
    expect(within(dropdown).getByAltText("")).toHaveAttribute("src", expect.stringContaining("anthropic.svg"));
  });

  it("should select a provider and reset the filter with All providers", async () => {
    const user = userEvent.setup();
    render(<ProviderFilter />);

    const dropdown = screen.getByRole("combobox", { name: "Provider" });
    expect(dropdown).toHaveTextContent("All providers");
    await chooseSelectOption(user, dropdown, "OpenAI");
    expect(dropdown).toHaveTextContent("OpenAI");
    await chooseSelectOption(user, dropdown, "All providers");
    expect(dropdown).toHaveTextContent("All providers");
    expect(within(dropdown).queryByAltText("")).not.toBeInTheDocument();
  });

  it("should disable the dropdown until the initial provider list arrives", () => {
    render(<ModelHubProviders providers={[]} selectedProviders={[]} onSelect={() => undefined} loading />);

    const dropdown = screen.getByRole("combobox", { name: "Provider" });
    expect(dropdown).toBeDisabled();
    expect(dropdown).toHaveTextContent("Loading providers");
  });

  it("should let keyboard users open the dropdown and select a provider", async () => {
    const user = userEvent.setup();
    render(<ProviderFilter />);

    const dropdown = screen.getByRole("combobox", { name: "Provider" });
    await user.tab();
    expect(dropdown).toHaveFocus();
    await user.keyboard("{ArrowDown}");
    expect(await screen.findByRole("option", { name: "Anthropic" })).toBeInTheDocument();
    await user.keyboard("{End}{Enter}");
    expect(dropdown).toHaveTextContent("Anthropic");
    expect(dropdown).toHaveFocus();
  });
});

describe("ProviderLogo", () => {
  it("should show a provider initial when its logo is unknown", () => {
    render(<ProviderLogo provider="local-models" />);

    expect(screen.getByText("L")).toBeInTheDocument();
    expect(screen.queryByAltText("")).not.toBeInTheDocument();
  });

  it("should fall back after a broken logo and load a different provider's logo", () => {
    const { rerender } = render(<ProviderLogo provider="openai" />);

    act(() => fireEvent.error(screen.getByAltText("")));
    expect(screen.getByText("O")).toBeInTheDocument();
    expect(screen.queryByAltText("")).not.toBeInTheDocument();

    rerender(<ProviderLogo provider="anthropic" />);
    expect(screen.getByAltText("")).toHaveAttribute("src", expect.stringContaining("anthropic.svg"));
    expect(screen.queryByText("O")).not.toBeInTheDocument();
  });
});
