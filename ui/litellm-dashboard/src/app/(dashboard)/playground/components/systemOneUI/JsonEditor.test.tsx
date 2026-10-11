import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import JsonEditor from "./JsonEditor";
import { validateSystemOnePayload } from "./lib/validatePayload";

const validPayload = JSON.stringify(
  { state: "Hi", questions: { escalate: { type: "noul", instructions: "Escalate?" } } },
  null,
  2,
);

describe("JsonEditor", () => {
  it("marks a valid payload as ready to send and counts its lines", () => {
    render(<JsonEditor value={validPayload} onChange={vi.fn()} validation={validateSystemOnePayload(validPayload)} />);

    expect(screen.getByText("Valid payload")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Ready to send");
    expect(screen.getByText(`${validPayload.split("\n").length} lines`)).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "System One JSON payload" })).toHaveAttribute("aria-invalid", "false");
  });

  it("lists each issue with its path and counts only errors in the status badge", () => {
    const payload = JSON.stringify({
      state: "Hi",
      questions: {
        category: { type: "choice", instructions: 1, criteria: { support: "Help" } },
        urgency: { type: "score", instructions: "Rate", criteria: ["Low"] },
      },
    });
    render(<JsonEditor value={payload} onChange={vi.fn()} validation={validateSystemOnePayload(payload)} />);

    expect(screen.getByText("2 issues")).toBeInTheDocument();
    const issues = screen.getByRole("list", { name: "Payload validation issues" });
    expect(issues).toHaveTextContent("questions.category.instructionsInstructions must be a string.");
    expect(issues).toHaveTextContent("questions.urgency.criteriaScore criteria must contain at least 2 levels.");
    expect(screen.getByRole("textbox", { name: "System One JSON payload" })).toHaveAttribute("aria-invalid", "true");
  });

  it("shows warnings without counting them as issues", () => {
    const payload = JSON.stringify({
      state: "Hi",
      questions: { urgency: { type: "score", instructions: "Rate", criteria: Array.from({ length: 11 }, () => "L") } },
    });
    render(<JsonEditor value={payload} onChange={vi.fn()} validation={validateSystemOnePayload(payload)} />);

    expect(screen.getByText("Valid payload")).toBeInTheDocument();
    expect(screen.getByRole("list", { name: "Payload validation issues" })).toHaveTextContent(
      "More than 10 score levels may reduce result quality.",
    );
  });

  it("numbers every line, including a trailing empty one, so wrapped lines keep their number", () => {
    const value = `${validPayload}\n`;
    render(<JsonEditor value={value} onChange={vi.fn()} validation={validateSystemOnePayload(value)} />);

    const lineCount = value.split("\n").length;
    expect(screen.getByText(`${lineCount} lines`)).toBeInTheDocument();
    expect(screen.getByText(String(lineCount))).toBeInTheDocument();
    expect(screen.queryByText(String(lineCount + 1))).not.toBeInTheDocument();
  });

  it("reports edits to the caller", () => {
    const onChange = vi.fn();
    render(<JsonEditor value={validPayload} onChange={onChange} validation={validateSystemOnePayload(validPayload)} />);

    fireEvent.change(screen.getByRole("textbox", { name: "System One JSON payload" }), { target: { value: "{}" } });

    expect(onChange).toHaveBeenCalledWith("{}");
  });
});
