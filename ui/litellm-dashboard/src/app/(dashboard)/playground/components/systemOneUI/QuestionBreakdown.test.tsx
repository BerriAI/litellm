import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import QuestionBreakdown from "./QuestionBreakdown";
import { validateSystemOnePayload } from "./lib/validatePayload";

const previewOf = (questions: object) => validateSystemOnePayload(JSON.stringify({ state: "Hi", questions })).payload;

describe("QuestionBreakdown", () => {
  it("previews every noul criterion that will be sent, not only true and false", () => {
    render(
      <QuestionBreakdown
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
    render(<QuestionBreakdown payload={previewOf({ escalate: { type: "noul", instructions: "Escalate?" } })} />);

    expect(screen.getByText("No criteria defined")).toBeInTheDocument();
  });
});
