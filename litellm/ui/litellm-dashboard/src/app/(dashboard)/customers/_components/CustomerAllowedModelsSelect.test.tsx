import { useState } from "react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "../../../../../tests/test-utils";
import CustomerAllowedModelsSelect from "./CustomerAllowedModelsSelect";

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAllProxyModels: () => ({
    data: {
      data: [
        { id: "openai/gpt-4.1", owned_by: "openai" },
        { id: "anthropic/claude-sonnet", owned_by: "anthropic" },
      ],
    },
    isLoading: false,
    isError: false,
  }),
}));

function Controlled({ initial = [] }: { initial?: string[] }) {
  const [value, setValue] = useState(initial);
  return <CustomerAllowedModelsSelect id="models" value={value} onChange={setValue} />;
}

describe("CustomerAllowedModelsSelect", () => {
  it("should render a searchable multiple selection using configured gateway models", async () => {
    const user = userEvent.setup();
    render(<Controlled />);
    const search = screen.getByRole("combobox", { name: "Search gateway models" });
    await user.type(search, "claude");
    await user.click(await screen.findByRole("option", { name: "anthropic/claude-sonnet" }));
    await user.type(search, "gpt");
    await user.click(await screen.findByRole("option", { name: "openai/gpt-4.1" }));
    expect(screen.getByLabelText("anthropic/claude-sonnet")).toBeInTheDocument();
    expect(screen.getByLabelText("openai/gpt-4.1")).toBeInTheDocument();
  });

  it("should preserve an unavailable selection while allowing its chip to be removed", async () => {
    const user = userEvent.setup();
    render(<Controlled initial={["retired-alias"]} />);
    const chip = screen.getByLabelText("retired-alias");
    await user.click(screen.getByRole("combobox", { name: "Search gateway models" }));
    expect(await screen.findByRole("option", { name: /retired-alias/ })).toHaveAttribute("aria-disabled", "true");
    await user.keyboard("{Escape}");
    await user.click(within(chip).getByRole("button"));
    expect(screen.queryByLabelText("retired-alias")).not.toBeInTheDocument();
  });

  it("should never offer a model that was only typed into the search", async () => {
    const user = userEvent.setup();
    render(<Controlled />);
    await user.type(screen.getByRole("combobox", { name: "Search gateway models" }), "unconfigured-model");
    expect(screen.queryByRole("option", { name: "unconfigured-model" })).not.toBeInTheDocument();
    expect(await screen.findByText("No options found")).toBeInTheDocument();
  });
});
