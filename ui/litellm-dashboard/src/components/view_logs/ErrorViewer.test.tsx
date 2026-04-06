import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { ErrorViewer } from "./ErrorViewer";

const basicError = {
  error_class: "NotFoundError",
  error_message: "Model gpt-5 not found",
};

const errorWithTraceback = {
  error_class: "AuthenticationError",
  error_message: "Invalid API key",
  traceback: `Traceback (most recent call last):
  File "/app/main.py", line 42, in handle_request
    result = await client.chat(model="gpt-4")
  File "/app/llms/openai.py", line 100, in chat
    response = self._make_request(payload)
  File "/app/llms/base.py", line 55, in _make_request
    raise AuthenticationError("Invalid API key")`,
};

describe("ErrorViewer", () => {
  it("should render error type and message", () => {
    render(<ErrorViewer errorInfo={basicError} />);
    expect(screen.getByText("NotFoundError")).toBeInTheDocument();
    expect(screen.getByText("Model gpt-5 not found")).toBeInTheDocument();
  });

  it("should show 'Unknown Error' when error_class is missing", () => {
    render(<ErrorViewer errorInfo={{ error_message: "something broke" }} />);
    expect(screen.getByText("Unknown Error")).toBeInTheDocument();
  });

  it("should show 'Unknown error occurred' when error_message is missing", () => {
    render(<ErrorViewer errorInfo={{ error_class: "RuntimeError" }} />);
    expect(screen.getByText("Unknown error occurred")).toBeInTheDocument();
  });

  it("should render traceback frames when traceback is present", () => {
    render(<ErrorViewer errorInfo={errorWithTraceback} />);
    expect(screen.getByText("Traceback")).toBeInTheDocument();
    expect(screen.getByText("main.py")).toBeInTheDocument();
    expect(screen.getByText("openai.py")).toBeInTheDocument();
    expect(screen.getByText("base.py")).toBeInTheDocument();
  });

  it("should expand a frame when clicked", async () => {
    const user = userEvent.setup();
    render(<ErrorViewer errorInfo={errorWithTraceback} />);

    await user.click(screen.getByText("main.py"));

    expect(
      screen.getByText("result = await client.chat(model=\"gpt-4\")")
    ).toBeInTheDocument();
  });

  it("should expand all frames when 'Expand All' is clicked", async () => {
    const user = userEvent.setup();
    render(<ErrorViewer errorInfo={errorWithTraceback} />);

    await user.click(screen.getByText("Expand All"));

    expect(screen.getByText("Collapse All")).toBeInTheDocument();
  });

  it("should copy traceback to clipboard when copy button is clicked", async () => {
    const user = userEvent.setup();
    const mockWriteText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { writeText: mockWriteText },
      writable: true,
      configurable: true,
    });

    render(<ErrorViewer errorInfo={errorWithTraceback} />);

    await user.click(screen.getByTitle("Copy traceback"));
    expect(mockWriteText).toHaveBeenCalledWith(errorWithTraceback.traceback);
  });

  it("should not render traceback section when traceback is absent", () => {
    render(<ErrorViewer errorInfo={basicError} />);
    expect(screen.queryByText("Traceback")).not.toBeInTheDocument();
  });
});
