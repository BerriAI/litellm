import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageActions from "./MessageActions";

describe("MessageActions", () => {
  const defaultProps = {
    role: "user",
    content: "Hello world",
    messageIndex: 0,
    isLastAssistantMessage: false,
    isLoading: false,
    onRetry: vi.fn(),
    onEditSubmit: vi.fn(),
  };

  it("should render edit button for user messages on hover", () => {
    render(<MessageActions {...defaultProps} />);
    const editButton = screen.getByTestId("edit-message-button");
    expect(editButton).toBeInTheDocument();
  });

  it("should not render any buttons for user messages when loading", () => {
    render(<MessageActions {...defaultProps} isLoading={true} />);
    expect(screen.queryByTestId("edit-message-button")).toBeNull();
    expect(screen.queryByTestId("retry-message-button")).toBeNull();
  });

  it("should render retry button for the last assistant message", () => {
    render(
      <MessageActions
        {...defaultProps}
        role="assistant"
        isLastAssistantMessage={true}
      />,
    );
    const retryButton = screen.getByTestId("retry-message-button");
    expect(retryButton).toBeInTheDocument();
  });

  it("should not render retry button for non-last assistant messages", () => {
    render(
      <MessageActions
        {...defaultProps}
        role="assistant"
        isLastAssistantMessage={false}
      />,
    );
    expect(screen.queryByTestId("retry-message-button")).toBeNull();
  });

  it("should not render edit button for assistant messages", () => {
    render(
      <MessageActions
        {...defaultProps}
        role="assistant"
        isLastAssistantMessage={true}
      />,
    );
    expect(screen.queryByTestId("edit-message-button")).toBeNull();
  });

  it("should not render any buttons for image messages", () => {
    render(<MessageActions {...defaultProps} isImage={true} />);
    expect(screen.queryByTestId("edit-message-button")).toBeNull();
    expect(screen.queryByTestId("message-actions")).toBeNull();
  });

  it("should not render any buttons for audio messages", () => {
    render(<MessageActions {...defaultProps} isAudio={true} />);
    expect(screen.queryByTestId("message-actions")).toBeNull();
  });

  it("should call onRetry when retry button is clicked", () => {
    const onRetry = vi.fn();
    render(
      <MessageActions
        {...defaultProps}
        role="assistant"
        isLastAssistantMessage={true}
        onRetry={onRetry}
      />,
    );

    act(() => {
      fireEvent.click(screen.getByTestId("retry-message-button"));
    });

    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("should show edit textarea when edit button is clicked", async () => {
    render(<MessageActions {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeInTheDocument();
    expect(textarea).toHaveValue("Hello world");
  });

  it("should call onEditSubmit when confirming edit", async () => {
    const onEditSubmit = vi.fn();
    render(<MessageActions {...defaultProps} onEditSubmit={onEditSubmit} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    act(() => {
      fireEvent.change(textarea, { target: { value: "Updated message" } });
    });

    act(() => {
      fireEvent.click(screen.getByTestId("confirm-edit-button"));
    });

    expect(onEditSubmit).toHaveBeenCalledWith(0, "Updated message");
  });

  it("should cancel edit when cancel button is clicked", async () => {
    render(<MessageActions {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    act(() => {
      fireEvent.click(screen.getByTestId("cancel-edit-button"));
    });

    await waitFor(() => {
      expect(screen.queryByTestId("edit-message-container")).toBeNull();
    });

    expect(screen.getByTestId("edit-message-button")).toBeInTheDocument();
  });

  it("should cancel edit when Escape key is pressed", async () => {
    render(<MessageActions {...defaultProps} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    act(() => {
      fireEvent.keyDown(textarea, { key: "Escape" });
    });

    await waitFor(() => {
      expect(screen.queryByTestId("edit-message-container")).toBeNull();
    });
  });

  it("should submit edit when Enter key is pressed without Shift", async () => {
    const onEditSubmit = vi.fn();
    render(<MessageActions {...defaultProps} onEditSubmit={onEditSubmit} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    act(() => {
      fireEvent.change(textarea, { target: { value: "New content" } });
    });

    act(() => {
      fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    });

    expect(onEditSubmit).toHaveBeenCalledWith(0, "New content");
  });

  it("should not submit edit with empty content", async () => {
    const onEditSubmit = vi.fn();
    render(<MessageActions {...defaultProps} onEditSubmit={onEditSubmit} />);

    act(() => {
      fireEvent.click(screen.getByTestId("edit-message-button"));
    });

    await waitFor(() => {
      expect(screen.getByTestId("edit-message-container")).toBeInTheDocument();
    });

    const textarea = screen.getByRole("textbox");
    act(() => {
      fireEvent.change(textarea, { target: { value: "   " } });
    });

    const confirmButton = screen.getByTestId("confirm-edit-button");
    expect(confirmButton).toBeDisabled();
  });
});
