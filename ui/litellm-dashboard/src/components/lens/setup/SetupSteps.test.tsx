import { useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { SetupStep as SetupStepId } from "./investigationSchema";
import { SetupStep, SetupSteps } from "./SetupSteps";

function Stepper({ initial }: { initial: SetupStepId }) {
  const [current, setCurrent] = useState<SetupStepId>(initial);
  return (
    <SetupSteps aria-label="Steps" current={current} onOpen={setCurrent}>
      <SetupStep id="activity" heading="Activity" description="Pick activity" summary="support_agent · Last day">
        <input aria-label="Agent" />
      </SetupStep>
      <SetupStep id="criteria" heading="Criteria" description="Describe checks" summary="2 checks">
        <textarea aria-label="Expectations" />
      </SetupStep>
      <SetupStep id="run" heading="Run" description="Schedule and budget" summary="">
        <button type="button">Run</button>
      </SetupStep>
    </SetupSteps>
  );
}

describe("SetupSteps", () => {
  it("shows the current step's fields, finished steps as summaries and later steps as descriptions", () => {
    render(<Stepper initial="criteria" />);
    const steps = within(screen.getByRole("list", { name: "Steps" }));
    const activity = steps.getByRole("button", { name: /^Activity/ });
    expect(activity).toBeEnabled();
    expect(activity).toHaveTextContent("support_agent · Last day");
    expect(activity).not.toHaveTextContent("Pick activity");
    const criteria = steps.getByRole("button", { name: /^Criteria/ });
    expect(criteria).toHaveAttribute("aria-current", "step");
    expect(criteria).toHaveTextContent("Describe checks");
    const run = steps.getByRole("button", { name: /^Run/ });
    expect(run).toBeDisabled();
    expect(run).toHaveTextContent("Schedule and budget");
    expect(screen.getByRole("textbox", { name: "Expectations" })).toBeVisible();
    expect(screen.queryByRole("textbox", { name: "Agent" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
  });

  it("reopens a finished step from its summary and collapses the one that was open", async () => {
    const user = userEvent.setup();
    render(<Stepper initial="run" />);
    await user.click(screen.getByRole("button", { name: /^Criteria/ }));
    expect(screen.getByRole("textbox", { name: "Expectations" })).toBeVisible();
    expect(screen.queryByRole("button", { name: "Run" })).not.toBeInTheDocument();
    const run = screen.getByRole("button", { name: /^Run/ });
    expect(run).toBeDisabled();
    expect(run).toHaveTextContent("Schedule and budget");
    expect(screen.getByRole("button", { name: /^Activity/ })).toHaveTextContent("support_agent · Last day");
  });
});
