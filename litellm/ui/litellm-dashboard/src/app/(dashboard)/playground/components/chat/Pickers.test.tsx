import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChecklistMenu, TargetPicker, type OptionGroup } from "./Pickers";

const GROUPS: readonly OptionGroup[] = [
  {
    label: "openai",
    options: [
      { value: "gpt-fast", label: "gpt-fast" },
      { value: "gpt-smart", label: "gpt-smart" },
    ],
  },
];

const renderTargetPicker = (onValueChange = vi.fn()) => {
  render(
    <TargetPicker
      label="Model"
      placeholder="Select a model"
      groups={GROUPS}
      value={null}
      onValueChange={onValueChange}
      isLoading={false}
      emptyText="No models"
      allowCustom
    />,
  );
  return onValueChange;
};

describe("TargetPicker", () => {
  it("selects the first match when Enter is pressed after searching", async () => {
    const user = userEvent.setup();
    const onValueChange = renderTargetPicker();

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.type(await screen.findByPlaceholderText("Search models"), "smart{Enter}");

    expect(onValueChange).toHaveBeenCalledWith("gpt-smart");
  });

  it("uses a typed model name that matches nothing when Enter is pressed", async () => {
    const user = userEvent.setup();
    const onValueChange = renderTargetPicker();

    await user.click(screen.getByRole("combobox", { name: "Model" }));
    await user.type(await screen.findByPlaceholderText("Search models"), "my-custom-model{Enter}");

    expect(onValueChange).toHaveBeenCalledWith("my-custom-model");
  });
});

describe("ChecklistMenu", () => {
  it("toggles the first match when Enter is pressed after searching", async () => {
    const user = userEvent.setup();
    const onValueChange = vi.fn();
    render(
      <ChecklistMenu
        label="Guardrails"
        icon={null}
        groups={[
          {
            label: "Guardrails",
            options: [
              { value: "pii-mask", label: "pii-mask" },
              { value: "toxicity", label: "toxicity" },
            ],
          },
        ]}
        value={["pii-mask"]}
        onValueChange={onValueChange}
        isLoading={false}
        emptyText="No guardrails"
      />,
    );

    await user.click(screen.getByRole("combobox", { name: "Guardrails" }));
    await user.type(await screen.findByPlaceholderText("Search guardrails"), "tox{Enter}");

    expect(onValueChange).toHaveBeenLastCalledWith(["pii-mask", "toxicity"]);
  });
});
