import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { ModelThrottlingEditor } from "./ModelThrottlingEditor";
import type { ModelThrottling } from "./modelThrottling";

function Editor({ maxBudget = 100 }: { maxBudget?: number }) {
  const [value, onChange] = useState<ModelThrottling | null>(null);
  return (
    <ModelThrottlingEditor
      value={value}
      onChange={onChange}
      maxBudget={maxBudget}
      availableModels={["premium", "economy"]}
    />
  );
}

describe("ModelThrottlingEditor", () => {
  it("explains the prerequisite when no final budget is set", () => {
    render(<Editor maxBudget={0} />);
    expect(screen.getByRole("switch", { name: "Model throttling" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText("Set a final budget to enable model throttling.")).toBeInTheDocument();
  });

  it("adds and removes thresholds and validates the final cap inline", async () => {
    const user = userEvent.setup();
    render(<Editor />);
    await user.click(screen.getByRole("switch", { name: "Model throttling" }));
    fireEvent.change(screen.getByLabelText("Spend reaches"), { target: { value: "100" } });
    expect(screen.getByLabelText("Spend reaches")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText("Must be below the final budget")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Spend reaches"), { target: { value: "60" } });
    await user.click(screen.getByRole("button", { name: "Add threshold" }));
    expect(screen.getAllByLabelText("Spend reaches")).toHaveLength(2);
    await user.click(screen.getByRole("button", { name: "Remove threshold 2" }));
    expect(screen.getByLabelText("Spend reaches")).toHaveValue(60);
    await user.click(screen.getByRole("switch", { name: "Model throttling" }));
    expect(screen.queryByLabelText("Spend reaches")).not.toBeInTheDocument();
  });
});
