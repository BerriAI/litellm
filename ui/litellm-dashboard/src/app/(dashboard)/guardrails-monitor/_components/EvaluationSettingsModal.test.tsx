import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { EvaluationSettingsModal } from "./EvaluationSettingsModal";

const mockFetchAvailableModels = vi.fn();
vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: (...args: unknown[]) => mockFetchAvailableModels(...args),
}));

const modelGroups = [{ model_group: "gpt-5.2" }, { model_group: "claude-sonnet-5" }];

const defaultProps = {
  open: true,
  onClose: vi.fn(),
  guardrailName: "pii-detector",
  accessToken: "test-token",
  onRunEvaluation: vi.fn(),
};

async function selectModel(user: ReturnType<typeof userEvent.setup>, label: string) {
  await user.click(screen.getByRole("combobox"));
  const options = await screen.findAllByText(label);
  await user.click(options[options.length - 1]);
}

describe("EvaluationSettingsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetchAvailableModels.mockResolvedValue(modelGroups);
  });

  it("should render nothing while closed", () => {
    render(<EvaluationSettingsModal {...defaultProps} open={false} />);
    expect(screen.queryByText("Evaluation Settings")).not.toBeInTheDocument();
  });

  it("should show the title and the guardrail-specific description when open", () => {
    render(<EvaluationSettingsModal {...defaultProps} />);
    expect(screen.getByText("Evaluation Settings")).toBeInTheDocument();
    expect(screen.getByText("Configure AI evaluation for pii-detector")).toBeInTheDocument();
  });

  it("should fall back to a generic description when no guardrail name is given", () => {
    render(<EvaluationSettingsModal {...defaultProps} guardrailName={undefined} />);
    expect(screen.getByText("Configure AI evaluation for re-running on logs")).toBeInTheDocument();
  });

  it("should prefill the prompt and the response schema with their defaults", () => {
    render(<EvaluationSettingsModal {...defaultProps} />);
    expect(screen.getByDisplayValue(/Evaluate whether this guardrail's decision was correct/)).toBeInTheDocument();
    expect(
      screen.getByDisplayValue(/"verdict": "correct" \| "false_positive" \| "false_negative"/),
    ).toBeInTheDocument();
  });

  it("should restore the default prompt when 'Reset to default' is clicked", async () => {
    const user = userEvent.setup();
    render(<EvaluationSettingsModal {...defaultProps} />);

    const promptBox = screen.getByDisplayValue(/Evaluate whether this guardrail's decision was correct/);
    await user.clear(promptBox);
    fireEvent.change(promptBox, { target: { value: "custom prompt" } });
    expect(screen.getByDisplayValue("custom prompt")).toBeInTheDocument();

    await user.click(screen.getByText("Reset to default"));
    expect(screen.getByDisplayValue(/Evaluate whether this guardrail's decision was correct/)).toBeInTheDocument();
  });

  it("should load the available models with the access token when opened", async () => {
    render(<EvaluationSettingsModal {...defaultProps} />);
    await waitFor(() => expect(mockFetchAvailableModels).toHaveBeenCalledWith("test-token"));
  });

  it("should not load models when there is no access token", () => {
    render(<EvaluationSettingsModal {...defaultProps} accessToken={null} />);
    expect(mockFetchAvailableModels).not.toHaveBeenCalled();
  });

  it("should not run an evaluation while no model is selected", async () => {
    const user = userEvent.setup();
    const onRunEvaluation = vi.fn();
    const onClose = vi.fn();
    render(<EvaluationSettingsModal {...defaultProps} onRunEvaluation={onRunEvaluation} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /run evaluation/i }));

    expect(onRunEvaluation).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("should run the evaluation with the selected model and the current prompt and schema", async () => {
    const user = userEvent.setup();
    const onRunEvaluation = vi.fn();
    const onClose = vi.fn();
    render(<EvaluationSettingsModal {...defaultProps} onRunEvaluation={onRunEvaluation} onClose={onClose} />);

    await waitFor(() => expect(mockFetchAvailableModels).toHaveBeenCalled());
    await selectModel(user, "claude-sonnet-5");
    await user.click(screen.getByRole("button", { name: /run evaluation/i }));

    expect(onRunEvaluation).toHaveBeenCalledWith({
      model: "claude-sonnet-5",
      prompt: expect.stringContaining("Evaluate whether this guardrail's decision was correct"),
      schema: expect.stringContaining('"verdict"'),
    });
    expect(onClose).toHaveBeenCalled();
  });

  it("should close without running when 'Cancel' is clicked", async () => {
    const user = userEvent.setup();
    const onRunEvaluation = vi.fn();
    const onClose = vi.fn();
    render(<EvaluationSettingsModal {...defaultProps} onRunEvaluation={onRunEvaluation} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /cancel/i }));

    expect(onClose).toHaveBeenCalled();
    expect(onRunEvaluation).not.toHaveBeenCalled();
  });
});
