import React from "react";
import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { SimpleMessageBlock } from "./SimpleMessageBlock";

describe("SimpleMessageBlock", () => {
  it("should render the label and content", () => {
    render(<SimpleMessageBlock label="USER" content="Hello world" />);
    expect(screen.getByText("USER")).toBeInTheDocument();
    expect(screen.getByText("Hello world")).toBeInTheDocument();
  });

  it("should return null when content is empty and no tool calls", () => {
    const { container } = render(
      <SimpleMessageBlock label="USER" content="" />
    );
    expect(container.innerHTML).toBe("");
  });

  it('should return null when content is "null" string and no tool calls', () => {
    const { container } = render(
      <SimpleMessageBlock label="USER" content="null" />
    );
    expect(container.innerHTML).toBe("");
  });

  it("should render tool calls when present", () => {
    render(
      <SimpleMessageBlock
        label="ASSISTANT"
        toolCalls={[
          { id: "tc1", name: "get_weather", arguments: { city: "Paris" } },
        ]}
      />
    );
    expect(screen.getByText("ASSISTANT")).toBeInTheDocument();
    expect(screen.getByText("get_weather")).toBeInTheDocument();
  });

  it("should render content and tool calls together", () => {
    render(
      <SimpleMessageBlock
        label="ASSISTANT"
        content="Let me check the weather."
        toolCalls={[
          { id: "tc1", name: "get_weather", arguments: { city: "Paris" } },
        ]}
      />
    );
    expect(
      screen.getByText("Let me check the weather.")
    ).toBeInTheDocument();
    expect(screen.getByText("get_weather")).toBeInTheDocument();
  });
});
