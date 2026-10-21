import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { IssueList, ValidationStatus } from "./ValidationFeedback";
import type { PayloadValidation } from "./lib/validatePayload";

const validation = (issues: PayloadValidation<unknown>["issues"]): PayloadValidation<unknown> => ({
  isValid: issues.every((issue) => issue.severity !== "error"),
  issues,
});

describe("ValidationStatus", () => {
  it("counts only errors as issues", () => {
    render(
      <ValidationStatus
        validation={validation([
          { path: "state", message: "Required", severity: "error" },
          { path: "model", message: "Required", severity: "error" },
          { path: "questions.a", message: "Long question", severity: "warning" },
        ])}
      />,
    );
    expect(screen.getByText("2 issues")).toBeInTheDocument();
  });

  it("reports a valid payload when only warnings remain", () => {
    render(<ValidationStatus validation={validation([{ path: "q", message: "Long", severity: "warning" }])} />);
    expect(screen.getByText("Valid payload")).toBeInTheDocument();
  });
});

describe("IssueList", () => {
  it("lists every issue with its path and says ready when there are none", () => {
    const { rerender } = render(
      <IssueList id="issues" validation={validation([{ path: "state", message: "Required", severity: "error" }])} />,
    );
    expect(screen.getByRole("list", { name: "Payload validation issues" })).toHaveTextContent("stateRequired");
    rerender(<IssueList id="issues" validation={validation([])} />);
    expect(screen.getByRole("status")).toHaveTextContent("Ready to send");
  });
});
