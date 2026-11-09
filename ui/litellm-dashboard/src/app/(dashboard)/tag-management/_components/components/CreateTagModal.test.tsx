import { fireEvent, render, screen } from "@testing-library/react";
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
    fireEvent.change(tagNameInput, { target: { value: "test-tag" } });

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

  it("submits TPM and RPM limits typed in the budget section", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    fireEvent.change(screen.getByLabelText("Tag Name"), { target: { value: "limited-tag" } });
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("TPM Limit"), { target: { value: "50000" } });
    fireEvent.change(screen.getByLabelText("RPM Limit"), { target: { value: "120" } });

    await user.click(screen.getByRole("button", { name: /Create Tag/i }));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ tag_name: "limited-tag", tpm_limit: "50000", rpm_limit: "120" }),
    );
  });

  it("drops TPM and RPM limits typed in a budget section that was collapsed again", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    fireEvent.change(screen.getByLabelText("Tag Name"), { target: { value: "limited-tag" } });
    const toggle = () => screen.getByRole("button", { name: /Budget & Rate Limits/ });
    await user.click(toggle());
    fireEvent.change(await screen.findByLabelText("RPM Limit"), { target: { value: "120" } });
    await user.click(toggle());

    await user.click(screen.getByRole("button", { name: /Create Tag/i }));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ tag_name: "limited-tag", rpm_limit: undefined }),
    );
  });

  it("rejects a fractional TPM limit", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    fireEvent.change(screen.getByLabelText("Tag Name"), { target: { value: "limited-tag" } });
    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    fireEvent.change(await screen.findByLabelText("TPM Limit"), { target: { value: "2.5" } });

    await user.click(screen.getByRole("button", { name: /Create Tag/i }));

    expect(await screen.findByText("Enter a whole number of 0 or more")).toBeInTheDocument();
    expect(mockOnSubmit).not.toHaveBeenCalled();
  });

  it("no longer tells users that tag rate limits are unsupported", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    await user.click(screen.getByRole("button", { name: /Budget & Rate Limits/ }));
    await screen.findByLabelText("TPM Limit");

    expect(screen.queryByText(/not currently supported/)).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /GitHub issue/ })).not.toBeInTheDocument();
  });

  it("still creates the tag when an invalid limit was left behind in a collapsed budget section", async () => {
    const user = userEvent.setup();
    render(<CreateTagModal {...defaultProps} />);

    fireEvent.change(screen.getByLabelText("Tag Name"), { target: { value: "limited-tag" } });
    const toggle = () => screen.getByRole("button", { name: /Budget & Rate Limits/ });
    await user.click(toggle());
    fireEvent.change(await screen.findByLabelText("TPM Limit"), { target: { value: "-1" } });
    await user.click(toggle());

    await user.click(screen.getByRole("button", { name: /Create Tag/i }));

    expect(mockOnSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ tag_name: "limited-tag", tpm_limit: undefined }),
    );
  });
});
