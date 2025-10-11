import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import CreateTagModal from "./CreateTagModal";

describe("CreateTagModal", () => {
  const mockOnCancel = vi.fn();
  const mockOnSubmit = vi.fn();
  const mockAvailableModels = [
    {
      model_name: "gpt-4",
      litellm_params: {
        model: "gpt-4",
      },
      model_info: {
        id: "model-123",
      },
    },
    {
      model_name: "claude-3",
      litellm_params: {
        model: "claude-3",
      },
      model_info: {
        id: "model-456",
      },
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should submit form with correct values when user creates a tag", async () => {
    /**
     * Tests that filling out the form and clicking submit calls onSubmit with the correct values.
     * This is the core functionality of the CreateTagModal component.
     */
    const user = userEvent.setup();

    render(
      <CreateTagModal
        visible={true}
        onCancel={mockOnCancel}
        onSubmit={mockOnSubmit}
        availableModels={mockAvailableModels}
      />
    );

    // Wait for modal to be visible
    await waitFor(() => {
      expect(screen.getByText("Create New Tag")).toBeInTheDocument();
    });

    // Fill in the tag name
    const tagNameInput = screen.getByLabelText("Tag Name");
    await user.type(tagNameInput, "production-tag");

    // Fill in the description
    const descriptionTextarea = screen.getByLabelText("Description");
    await user.type(descriptionTextarea, "Tag for production environment");

    // Submit the form
    const createButton = screen.getByText("Create Tag");
    await user.click(createButton);

    // Verify onSubmit was called with correct values
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          tag_name: "production-tag",
          description: "Tag for production environment",
        })
      );
    });

    // Verify onCancel was not called
    expect(mockOnCancel).not.toHaveBeenCalled();
  });
});

