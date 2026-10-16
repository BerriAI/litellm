import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FormProvider, useForm, useWatch } from "react-hook-form";
import { describe, expect, it } from "vitest";
import LLMJudgeFields from "./LLMJudgeFields";
import type { GuardrailFormValues } from "../GuardrailFormField";

const Probe = () => {
  const form = useForm<GuardrailFormValues>({
    defaultValues: { criteria: [{ name: "", weight: 100, description: "" }] },
  });
  const sampleRate = useWatch({ control: form.control, name: "sample_rate" });
  return (
    <FormProvider {...form}>
      <output data-testid="sample-rate-state">{String(sampleRate)}</output>
      <LLMJudgeFields availableModels={[]} control={form.control} />
    </FormProvider>
  );
};

// jsdom sanitizes input[type=number].value, so an in-progress "0." can never
// reach the DOM; the states below are the contract the payload builder relies on.
describe("LLMJudgeFields", () => {
  it("keeps the Sample rate state as the raw typed text", async () => {
    const user = userEvent.setup({ delay: null });
    render(<Probe />);

    await user.type(screen.getByLabelText(/Sample rate/), "0.05");
    expect(screen.getByTestId("sample-rate-state")).toHaveTextContent("0.05");
  });

  it("clears the Sample rate state to null when the input is emptied", async () => {
    const user = userEvent.setup({ delay: null });
    render(<Probe />);

    const sampleRate = screen.getByLabelText(/Sample rate/);
    await user.type(sampleRate, "0.5");
    await user.clear(sampleRate);
    expect(screen.getByTestId("sample-rate-state")).toHaveTextContent("null");
  });
});
