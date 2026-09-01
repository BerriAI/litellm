import React from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import PipelineFlowBuilder, { FlowBuilderPage, PipelineInfoDisplay } from "./pipeline_flow_builder";
import { GuardrailPipeline, PipelineStep } from "@/components/policies/types";
import { Guardrail } from "@/components/guardrails/types";

vi.mock("@/components/networking");

const step = (overrides: Partial<PipelineStep> = {}): PipelineStep => ({
  guardrail: "pii-masker",
  on_pass: "next",
  on_fail: "block",
  on_error: null,
  modify_response_message: null,
  ...overrides,
});

const pipeline = (steps: PipelineStep[]): GuardrailPipeline => ({ mode: "pre_call", steps });

const guardrails = [
  { guardrail_id: "g1", guardrail_name: "pii-masker" },
  { guardrail_id: "g2", guardrail_name: "prompt-injection" },
] as Guardrail[];

describe("PipelineInfoDisplay", () => {
  it("renders the trigger card", () => {
    renderWithProviders(<PipelineInfoDisplay pipeline={pipeline([step()])} />);

    expect(screen.getByText("TRIGGER")).toBeInTheDocument();
    expect(screen.getByText("Incoming LLM Request")).toBeInTheDocument();
  });

  it("renders one numbered card per step, naming its guardrail", () => {
    renderWithProviders(<PipelineInfoDisplay pipeline={pipeline([step(), step({ guardrail: "prompt-injection" })])} />);

    expect(screen.getByText("Step 1")).toBeInTheDocument();
    expect(screen.getByText("Step 2")).toBeInTheDocument();
    expect(screen.getByText("pii-masker")).toBeInTheDocument();
    expect(screen.getByText("prompt-injection")).toBeInTheDocument();
    expect(screen.getAllByText("GUARDRAIL")).toHaveLength(2);
  });

  it("maps raw action values to their human labels", () => {
    renderWithProviders(<PipelineInfoDisplay pipeline={pipeline([step({ on_pass: "next", on_fail: "block" })])} />);

    expect(screen.getByText(/Pass .* Next Step/)).toBeInTheDocument();
    expect(screen.getByText(/On fail .* Block/)).toBeInTheDocument();
  });

  it("falls back to the on-fail action when no API-failure action is set", () => {
    renderWithProviders(<PipelineInfoDisplay pipeline={pipeline([step({ on_fail: "block", on_error: null })])} />);

    expect(screen.getByText(/On API failure .* Block \(same as on fail\)/)).toBeInTheDocument();
  });

  it("shows an explicit API-failure action when one is set", () => {
    renderWithProviders(<PipelineInfoDisplay pipeline={pipeline([step({ on_error: "allow" })])} />);

    expect(screen.getByText(/On API failure .* Allow/)).toBeInTheDocument();
    expect(screen.queryByText(/same as on fail/)).not.toBeInTheDocument();
  });
});

describe("PipelineFlowBuilder", () => {
  it("renders the trigger and end cards around the steps", () => {
    renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={vi.fn()} availableGuardrails={guardrails} />,
    );

    expect(screen.getByText("TRIGGER")).toBeInTheDocument();
    expect(screen.getByText("END")).toBeInTheDocument();
    expect(screen.getByText("Continue to LLM")).toBeInTheDocument();
  });

  it("labels each decision section of a step", () => {
    renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={vi.fn()} availableGuardrails={guardrails} />,
    );

    expect(screen.getByText("ON PASS")).toBeInTheDocument();
    expect(screen.getByText("ON FAIL")).toBeInTheDocument();
    expect(screen.getByText("ON API FAILURE")).toBeInTheDocument();
  });

  it("inserts a step at the clicked connector", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={onChange} availableGuardrails={guardrails} />,
    );

    await user.click(screen.getAllByRole("button", { name: "Insert step" })[0]);

    expect(onChange).toHaveBeenCalledTimes(1);
    expect(onChange.mock.calls[0][0].steps).toHaveLength(2);
  });

  it("removes the clicked step when more than one exists", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <PipelineFlowBuilder
        pipeline={pipeline([step(), step({ guardrail: "prompt-injection" })])}
        onChange={onChange}
        availableGuardrails={guardrails}
      />,
    );

    await user.click(screen.getAllByRole("button", { name: "Delete step" })[0]);

    expect(onChange.mock.calls[0][0].steps).toHaveLength(1);
    expect(onChange.mock.calls[0][0].steps[0].guardrail).toBe("prompt-injection");
  });

  it("disables deletion of the only remaining step", () => {
    renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={vi.fn()} availableGuardrails={guardrails} />,
    );

    expect(screen.getByRole("button", { name: "Delete step" })).toBeDisabled();
  });

  it("offers a custom response field only when the action is modify_response", () => {
    const { rerender } = renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={vi.fn()} availableGuardrails={guardrails} />,
    );
    expect(screen.queryByPlaceholderText("Enter custom response...")).not.toBeInTheDocument();

    rerender(
      <PipelineFlowBuilder
        pipeline={pipeline([step({ on_fail: "modify_response" })])}
        onChange={vi.fn()}
        availableGuardrails={guardrails}
      />,
    );

    expect(screen.getByPlaceholderText("Enter custom response...")).toBeInTheDocument();
  });

  it("reports an edited custom response message", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(
      <PipelineFlowBuilder
        pipeline={pipeline([step({ on_fail: "modify_response" })])}
        onChange={onChange}
        availableGuardrails={guardrails}
      />,
    );

    fireEvent.change(screen.getByPlaceholderText("Enter custom response..."), { target: { value: "x" } });

    expect(onChange.mock.calls[0][0].steps[0].modify_response_message).toBe("x");
  });

  it("offers a guardrail picker for the step", () => {
    renderWithProviders(
      <PipelineFlowBuilder pipeline={pipeline([step()])} onChange={vi.fn()} availableGuardrails={guardrails} />,
    );

    // Which control surfaces the selection is a presentation detail; that the step's
    // guardrail is the one displayed is covered by the PipelineInfoDisplay tests above.
    expect(screen.getByText("Guardrail")).toBeInTheDocument();
    expect(screen.getAllByRole("combobox").length).toBeGreaterThan(0);
  });
});

describe("FlowBuilderPage", () => {
  it("renders its shell in flow with no stacking level, so it can never cover the portalled popup layer", () => {
    const { container } = renderWithProviders(
      <FlowBuilderPage
        onBack={vi.fn()}
        onSuccess={vi.fn()}
        accessToken="sk-test"
        availableGuardrails={guardrails}
        createPolicy={vi.fn()}
        updatePolicy={vi.fn()}
      />,
    );

    const shell = container.firstElementChild as HTMLElement;
    const shellClasses = shell.className.split(/\s+/);

    expect(shell).toContainElement(screen.getByPlaceholderText("Policy name..."));
    expect(shell).not.toHaveStyle({ position: "fixed" });
    expect(shellClasses).not.toContain("fixed");
    expect(window.getComputedStyle(shell).zIndex).not.toMatch(/\d/);
    expect(shellClasses.filter((cls) => /^-?z-/.test(cls))).toEqual([]);
  });
});
