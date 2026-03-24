import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PipelineTestPanel } from "./pipeline_test_drawer";
import type { GuardrailPipeline, PipelineTestResult } from "./types";
import * as networking from "../networking";

vi.mock("../networking");
vi.mock("../../data/compliancePrompts", () => ({
  getFrameworks: () => [],
  getComplianceDatasetPrompts: () => [],
}));

const validPipeline: GuardrailPipeline = {
  mode: "pre_call",
  steps: [
    {
      guardrail: "content-filter",
      on_pass: "allow",
      on_fail: "block",
      pass_data: false,
      modify_response_message: null,
    },
  ],
};

const emptyGuardrailPipeline: GuardrailPipeline = {
  mode: "pre_call",
  steps: [
    {
      guardrail: "",
      on_pass: "next",
      on_fail: "block",
      pass_data: false,
      modify_response_message: null,
    },
  ],
};

const mockTestResult: PipelineTestResult = {
  terminal_action: "allow",
  step_results: [
    {
      guardrail_name: "content-filter",
      outcome: "pass",
      action_taken: "allow",
      modified_data: null,
      error_detail: null,
      duration_seconds: 0.123,
    },
  ],
  modified_data: null,
  error_message: null,
  modify_response_message: null,
};

const defaultProps = {
  pipeline: validPipeline,
  accessToken: "test-token",
  onClose: vi.fn(),
};

describe("PipelineTestPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);
    expect(screen.getByText("Test Pipeline")).toBeInTheDocument();
  });

  it("should call onClose when the close button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: /x/i }));
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it("should show the default test message textarea", () => {
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);
    expect(screen.getByPlaceholderText("Enter a test message...")).toBeInTheDocument();
  });

  it("should show placeholder text before any test is run", () => {
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);
    expect(screen.getByText(/choose a test source above/i)).toBeInTheDocument();
  });

  it("should show an error when a step has no guardrail selected", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <PipelineTestPanel
        {...defaultProps}
        pipeline={emptyGuardrailPipeline}
      />
    );
    await user.click(screen.getByRole("button", { name: /run test/i }));
    expect(screen.getByText("All steps must have a guardrail selected")).toBeInTheDocument();
  });

  it("should display step results after a successful test", async () => {
    vi.mocked(networking.testPipelineCall).mockResolvedValue(mockTestResult);
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /run test/i }));

    await waitFor(() => {
      expect(screen.getByText(/Step 1: content-filter/)).toBeInTheDocument();
    });
    expect(screen.getByText("PASS")).toBeInTheDocument();
    expect(screen.getByText(/123ms/)).toBeInTheDocument();
  });

  it("should display the terminal action after a successful test", async () => {
    vi.mocked(networking.testPipelineCall).mockResolvedValue(mockTestResult);
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /run test/i }));

    await waitFor(() => {
      expect(screen.getByText("allow")).toBeInTheDocument();
    });
  });

  it("should display 'Custom Response' for modify_response terminal action", async () => {
    const modifyResult: PipelineTestResult = {
      ...mockTestResult,
      terminal_action: "modify_response",
      modify_response_message: "Sorry, I can't help with that.",
    };
    vi.mocked(networking.testPipelineCall).mockResolvedValue(modifyResult);
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /run test/i }));

    await waitFor(() => {
      expect(screen.getByText("Custom Response")).toBeInTheDocument();
    });
    expect(screen.getByText(/Sorry, I can't help with that/)).toBeInTheDocument();
  });

  it("should display an error when the API call fails", async () => {
    vi.mocked(networking.testPipelineCall).mockRejectedValue(new Error("Network error"));
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /run test/i }));

    await waitFor(() => {
      expect(screen.getByText("Network error")).toBeInTheDocument();
    });
  });

  it("should not call the API when accessToken is null", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <PipelineTestPanel {...defaultProps} accessToken={null} />
    );

    await user.click(screen.getByRole("button", { name: /run test/i }));

    expect(networking.testPipelineCall).not.toHaveBeenCalled();
  });

  it("should show step error detail when present", async () => {
    const resultWithError: PipelineTestResult = {
      terminal_action: "block",
      step_results: [
        {
          guardrail_name: "content-filter",
          outcome: "error",
          action_taken: "block",
          modified_data: null,
          error_detail: "Guardrail timed out",
          duration_seconds: null,
        },
      ],
      modified_data: null,
      error_message: null,
      modify_response_message: null,
    };
    vi.mocked(networking.testPipelineCall).mockResolvedValue(resultWithError);
    const user = userEvent.setup();
    renderWithProviders(<PipelineTestPanel {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /run test/i }));

    await waitFor(() => {
      expect(screen.getByText("Guardrail timed out")).toBeInTheDocument();
    });
  });
});
