import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type React from "react";
import { MarkdownImage } from "./MarkdownImage";

const REMOTE_SRC = "http://attacker.example:4444/chart.png?token=secret";

describe("MarkdownImage", () => {
  it("renders a placeholder naming the host instead of an image until clicked", () => {
    render(<MarkdownImage src={REMOTE_SRC} alt="Weekly chart" />);

    expect(screen.queryByRole("img")).not.toBeInTheDocument();
    const load = screen.getByRole("button", { name: "Weekly chart attacker.example:4444 Load image" });
    expect(load).toHaveAttribute("title", REMOTE_SRC);

    fireEvent.click(load);

    expect(screen.getByRole("img", { name: "Weekly chart" })).toHaveAttribute("src", REMOTE_SRC);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("loads the image instead of following a link that wraps it", () => {
    const onAnchorClick = vi.fn((event: React.MouseEvent<HTMLAnchorElement>) => event.defaultPrevented);
    render(
      <a href="https://link.example/" onClick={onAnchorClick}>
        <MarkdownImage src={REMOTE_SRC} alt="Weekly chart" />
      </a>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Weekly chart attacker.example:4444 Load image" }));

    expect(screen.getByRole("img", { name: "Weekly chart" })).toHaveAttribute("src", REMOTE_SRC);
    expect(onAnchorClick).toHaveReturnedWith(true);
  });

  it("falls back to the raw source when it has no host", () => {
    render(<MarkdownImage src="/static/logo.png" alt="" />);
    render(<MarkdownImage src="mailto:ops@example.com" alt="" />);

    expect(screen.getByRole("button", { name: "Image /static/logo.png Load image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Image mailto:ops@example.com Load image" })).toBeInTheDocument();
  });

  it("renders only the alt text when there is no source", () => {
    render(<MarkdownImage alt="Weekly chart" />);

    expect(screen.getByText("Weekly chart")).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });
});
