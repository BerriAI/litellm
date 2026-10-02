import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import SystemOneQuestionBreakdown from "./SystemOneQuestionBreakdown";
import { validateSystemOnePayload } from "./validate_system_one_payload";

const previewOf = (questions: object) => validateSystemOnePayload(JSON.stringify({ state: "Hi", questions })).payload;

describe("SystemOneQuestionBreakdown", () => {
  it("previews every noul criterion that will be sent, not only true and false", () => {
    render(
      <SystemOneQuestionBreakdown
        payload={previewOf({
          escalate: {
            type: "noul",
            instructions: "Escalate?",
            criteria: { true: "Asks for a person", unsure: "Maybe" },
          },
        })}
      />,
    );

    expect(screen.getByText("Asks for a person")).toBeInTheDocument();
    expect(screen.getByText("unsure")).toBeInTheDocument();
    expect(screen.getByText("Maybe")).toBeInTheDocument();
    expect(screen.queryByText("No criteria defined")).not.toBeInTheDocument();
  });

  it("says when a noul question has no criteria", () => {
    render(
      <SystemOneQuestionBreakdown payload={previewOf({ escalate: { type: "noul", instructions: "Escalate?" } })} />,
    );

    expect(screen.getByText("No criteria defined")).toBeInTheDocument();
  });
});
