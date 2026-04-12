import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TruncatedValue } from "./TruncatedValue";

describe("TruncatedValue", () => {
  it("should render the value text", () => {
    render(<TruncatedValue value="chatcmpl-abc123" />);
    expect(screen.getByText("chatcmpl-abc123")).toBeInTheDocument();
  });

  it("should render a dash when value is undefined", () => {
    render(<TruncatedValue />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should render a dash when value is empty string", () => {
    render(<TruncatedValue value="" />);
    expect(screen.getByText("-")).toBeInTheDocument();
  });

  it("should apply the default maxWidth when not specified", () => {
    render(<TruncatedValue value="some-long-id-value" />);
    const el = screen.getByText("some-long-id-value");
    expect(el).toHaveStyle({ maxWidth: "180px" });
  });

  it("should apply custom maxWidth when provided", () => {
    render(<TruncatedValue value="test-value" maxWidth={300} />);
    const el = screen.getByText("test-value");
    expect(el).toHaveStyle({ maxWidth: "300px" });
  });
});
