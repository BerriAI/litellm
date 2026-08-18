import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { MessageInput } from "./MessageInput";

const PLACEHOLDER = "Type your message... (Shift+Enter for new line)";

describe("MessageInput", () => {
  it("renders a message box and a send control", () => {
    render(<MessageInput value="" onChange={vi.fn()} onSend={vi.fn()} />);

    expect(screen.getByPlaceholderText(PLACEHOLDER)).toBeInTheDocument();
    expect(screen.getByRole("button")).toBeInTheDocument();
  });

  it("reports every keystroke through onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<MessageInput value="" onChange={onChange} onSend={vi.fn()} />);

    await user.type(screen.getByPlaceholderText(PLACEHOLDER), "hi");

    expect(onChange).toHaveBeenCalledWith("h");
  });

  it("refuses to send while the box is empty and nothing is attached", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<MessageInput value="" onChange={vi.fn()} onSend={onSend} />);

    const send = screen.getByRole("button");
    expect(send).toBeDisabled();

    await user.click(send);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("sends once the box holds text", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<MessageInput value="hello" onChange={vi.fn()} onSend={onSend} />);

    const send = screen.getByRole("button");
    expect(send).toBeEnabled();

    await user.click(send);
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("sends on an attachment alone and surfaces the upload control", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(
      <MessageInput
        value=""
        onChange={vi.fn()}
        onSend={onSend}
        hasAttachment
        uploadComponent={<span data-testid="upload-component">Upload</span>}
      />,
    );

    expect(screen.getByTestId("upload-component")).toBeInTheDocument();

    const send = screen.getByRole("button");
    expect(send).toBeEnabled();

    await user.click(send);
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("stays inert while disabled even with text present", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<MessageInput value="hello" onChange={vi.fn()} onSend={onSend} disabled />);

    await user.click(screen.getByRole("button"));
    expect(onSend).not.toHaveBeenCalled();
  });

  it("sends on Enter but adds a newline on Shift+Enter", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<MessageInput value="hello" onChange={vi.fn()} onSend={onSend} />);

    const box = screen.getByPlaceholderText(PLACEHOLDER);
    await user.click(box);

    await user.keyboard("{Shift>}{Enter}{/Shift}");
    expect(onSend).not.toHaveBeenCalled();

    await user.keyboard("{Enter}");
    expect(onSend).toHaveBeenCalledTimes(1);
  });
});
