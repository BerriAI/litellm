import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MessageBubble from "./MessageBubble";

vi.mock("@/components/chat_ui/ResponseMetrics", () => ({ default: () => <div>metrics</div> }));

describe("MessageBubble", () => {
  it("renders assistant markdown and model", () => {
    render(<MessageBubble message={{ role: "assistant", content: "**Hello**", model: "gpt-4o" }} />);
    expect(screen.getByText("Hello")).toBeInTheDocument();
    expect(screen.getByText("gpt-4o")).toBeInTheDocument();
  });
});
