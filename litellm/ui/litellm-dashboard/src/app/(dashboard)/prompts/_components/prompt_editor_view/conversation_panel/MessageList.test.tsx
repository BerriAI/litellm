import { createRef } from "react";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import MessageList from "./MessageList";

describe("MessageList", () => {
  it("renders conversation messages", () => {
    render(
      <MessageList
        messages={[{ role: "user", content: "Hello" }]}
        isLoading={false}
        hasVariables={false}
        messagesEndRef={createRef<HTMLDivElement>()}
      />,
    );
    expect(screen.getByText("Hello")).toBeInTheDocument();
  });
});
