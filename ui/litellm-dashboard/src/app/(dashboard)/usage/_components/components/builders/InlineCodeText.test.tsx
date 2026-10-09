import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { InlineCodeText } from "./InlineCodeText";

describe("InlineCodeText", () => {
  it("renders backtick-delimited segments as styled code", () => {
    render(
      <p>
        <InlineCodeText text="Uses `codex` with `gpt-6-astra`." />
      </p>,
    );

    expect(screen.getByText("codex")).toHaveClass("rounded", "bg-muted", "px-1", "py-0.5", "font-mono", "text-xs");
    expect(screen.getByText("gpt-6-astra").tagName).toBe("CODE");
  });
});
