import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React, { useState } from "react";
import { describe, expect, it } from "vitest";

import {
  ModelMaxBudgetEditor,
  ModelMaxBudgetValue,
  modelMaxBudgetValueFromEntries,
  modelMaxBudgetEntriesFromValue,
} from "./ModelMaxBudgetEditor";

function ControlledEditor(props: { modelOptions: string[] }) {
  const [value, setValue] = useState<ModelMaxBudgetValue | null>(null);
  return <ModelMaxBudgetEditor {...props} value={value} onChange={setValue} />;
}

describe("ModelMaxBudgetEditor", () => {
  it("disables add when no model options are available", () => {
    render(<ModelMaxBudgetEditor modelOptions={[]} />);

    expect(screen.getByTestId("add-model-budget-button")).toBeDisabled();
  });

  it("adds a budget row when model options exist", async () => {
    const user = userEvent.setup({ delay: null });

    render(
      <ControlledEditor modelOptions={["claude-sonnet-4-6", "claude-opus-4-8"]} />,
    );

    const addButton = screen.getByTestId("add-model-budget-button");
    expect(addButton).toBeEnabled();

    await user.click(addButton);

    expect(screen.getByText("claude-sonnet-4-6")).toBeInTheDocument();
  });

  it("round-trips value through entry helpers", () => {
    const value = {
      "claude-sonnet-4-6": { budget_limit: 20, time_period: "1d" },
    };

    expect(modelMaxBudgetValueFromEntries(modelMaxBudgetEntriesFromValue(value))).toEqual(value);
  });
});
