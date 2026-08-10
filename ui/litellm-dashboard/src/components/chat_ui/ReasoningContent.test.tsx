import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import ReasoningContent from "./ReasoningContent";

vi.mock("react-i18next", async () => {
  const { resources } = await import("@/i18n/catalog");
  return {
    useTranslation: () => ({
      t: (key: string) =>
        key.split(".").reduce<unknown>((copy, segment) => {
          if (typeof copy !== "object" || copy === null) return undefined;
          return (copy as Record<string, unknown>)[segment];
        }, resources.en.chat) ?? key,
    }),
  };
});

describe("ReasoningContent", () => {
  it("should render nothing when reasoningContent is empty", () => {
    const { container } = render(<ReasoningContent reasoningContent="" />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should show reasoning content expanded by default and toggle on click", () => {
    render(<ReasoningContent reasoningContent="thinking hard" />);

    expect(screen.getByText("thinking hard")).toBeInTheDocument();
    expect(screen.getByText("Hide reasoning")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button"));

    expect(screen.queryByText("thinking hard")).not.toBeInTheDocument();
    expect(screen.getByText("Show reasoning")).toBeInTheDocument();
  });

  it("should constrain width and break long words so it cannot expand the layout (regression #32481)", () => {
    const longToken = "a".repeat(500);
    render(<ReasoningContent reasoningContent={longToken} />);

    const contentBox = screen.getByText(longToken).closest("div.mt-2");
    expect(contentBox).not.toBeNull();
    expect(contentBox).toHaveClass("max-w-full");
    expect(contentBox).toHaveStyle({ wordBreak: "break-word", overflowWrap: "break-word" });
  });
});
