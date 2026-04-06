import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MessageInput } from "./MessageInput";

describe("MessageInput", () => {
  it("should render", () => {
    const onChange = vi.fn();
    const onSend = vi.fn();
    const { container } = render(<MessageInput value="" onChange={onChange} onSend={onSend} />);
    const textarea = container.querySelector("textarea");
    const button = container.querySelector("button");
    expect(textarea).toBeInTheDocument();
    expect(button).toBeInTheDocument();
  });

  it("should disable send button initially", () => {
    const onChange = vi.fn();
    const onSend = vi.fn();
    const { container } = render(<MessageInput value="" onChange={onChange} onSend={onSend} />);
    const button = container.querySelector("button") as HTMLButtonElement;

    expect(button).toBeDisabled();
  });

  it("should enable send button when hasAttachment is true even with empty value", () => {
    const onChange = vi.fn();
    const onSend = vi.fn();
    const uploadComponent = <div data-testid="upload-component">Upload</div>;
    const { container, getByTestId } = render(
      <MessageInput value="" onChange={onChange} onSend={onSend} hasAttachment={true} uploadComponent={uploadComponent} />,
    );
    const button = container.querySelector("button") as HTMLButtonElement;
    expect(getByTestId("upload-component")).toBeInTheDocument();
    expect(button).not.toBeDisabled();
  });
});
