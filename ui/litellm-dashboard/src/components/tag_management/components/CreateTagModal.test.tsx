import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CreateTagModal from "./CreateTagModal";

describe("CreateTagModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnSubmit = vi.fn();
  const mockAvailableModels = [
    {
      model_name: "GPT-4",
      litellm_params: { model: "gpt-4" },
      model_info: { id: "model-1" },
    },
    {
      model_name: "Claude-3",
      litellm_params: { model: "claude-3" },
      model_info: { id: "model-2" },
    },
  ];

  const defaultProps = {
    visible: true,
    onCancel: mockOnCancel,
    onSubmit: mockOnSubmit,
    availableModels: mockAvailableModels,
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render the modal", () => {
    render(<CreateTagModal {...defaultProps} />);
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    expect(screen.getByText("Create New Tag")).toBeInTheDocument();
  });

  it("should submit form with required tag name", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    const tagNameInput = screen.getByLabelText("Tag Name");
    await user.type(tagNameInput, "test-tag");

    const submitButton = screen.getByRole("button", { name: /Create Tag/i });
    await user.click(submitButton);

    expect(mockOnSubmit).toHaveBeenCalledWith({
      tag_name: "test-tag",
    });
  });

  it("should not submit form when tag name is missing", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    const submitButton = screen.getByRole("button", { name: /Create Tag/i });
    await user.click(submitButton);

    // Form validation should prevent submission
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });
});
