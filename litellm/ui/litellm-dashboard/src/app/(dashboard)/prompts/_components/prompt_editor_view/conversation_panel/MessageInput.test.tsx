import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageInput from "./MessageInput";

describe("MessageInput", () => {
  it("updates and sends a message", () => {
    const onInputChange = vi.fn();
    const onSend = vi.fn();
    render(
      <MessageInput
        inputMessage="Hello"
        isLoading={false}
        isDisabled={false}
        onInputChange={onInputChange}
        onSend={onSend}
        onKeyDown={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    const textarea = screen.getByPlaceholderText(/type your message/i);
    expect(textarea).toHaveClass("field-sizing-content", "max-h-24", "overflow-y-auto");
    fireEvent.change(textarea, { target: { value: "Hi" } });
    fireEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(onInputChange).toHaveBeenCalledWith("Hi");
    expect(onSend).toHaveBeenCalledOnce();
  });
});
