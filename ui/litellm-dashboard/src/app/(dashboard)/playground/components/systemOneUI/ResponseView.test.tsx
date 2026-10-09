import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ResponseView from "./ResponseView";
import type { SystemOneResponse } from "./lib/schemas";

const sampleResponse: SystemOneResponse = {
  model: "jev-1.13.0",
  answers: {
    human: { type: "noul", noul: 0.99 },
    dept: {
      type: "choice",
      choice: "technical",
      confidence: 0.27,
      probabilities: { billing: 0.36, technical: 0.64 },
    },
    frustration: {
      type: "score",
      score: 2,
      confidence: 0.99,
      legend: { "0": "calm", "1": "mild", "2": "angry" },
      probabilities: { "0": 0, "1": 0, "2": 1 },
    },
  },
  usage: { input_tokens: 378, output_tokens: 63 },
};

describe("ResponseView", () => {
  it("renders calibrated probabilities for choice, noul, and score answers", () => {
    render(<ResponseView response={sampleResponse} isLoading={false} latencyMs={120} />);

    expect(screen.getAllByText("technical")).toHaveLength(2);
    const meters = screen.getAllByRole("meter");
    expect(meters.map((meter) => meter.getAttribute("aria-label"))).toEqual([
      "Yes probability",
      "technical probability",
      "billing probability",
      "0: calm probability",
      "1: mild probability",
      "2: angry probability",
    ]);
    expect(screen.getByRole("meter", { name: "technical probability" })).toHaveAttribute("aria-valuenow", "64");
    expect(screen.getByRole("meter", { name: "billing probability" })).toHaveAttribute("aria-valuenow", "36");
    expect(screen.getByText("99% yes")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Yes probability" })).toHaveAttribute("aria-valuenow", "99");
    expect(screen.getByRole("meter", { name: "2: angry probability" })).toBeInTheDocument();
    expect(screen.getByText("378 input / 63 output tokens")).toBeInTheDocument();
  });

  it("selects the score level nearest a fractional calibrated score", () => {
    const fractional: SystemOneResponse = {
      answers: {
        severity: { type: "score", score: 2.39, probabilities: { "1": 0, "2": 0.62, "3": 0.38 } },
      },
    };
    render(<ResponseView response={fractional} isLoading={false} />);

    expect(screen.getByRole("meter", { name: "2 probability" })).toHaveAttribute("aria-valuetext", "62%, selected");
    expect(screen.getByRole("meter", { name: "3 probability" })).toHaveAttribute("aria-valuetext", "38%");
    expect(screen.getByRole("meter", { name: "1 probability" })).toHaveAttribute("aria-valuetext", "0%");
  });

  it("renders structured decision score legends without coercing objects to strings", () => {
    const response: SystemOneResponse = {
      answers: {
        severity: {
          type: "score",
          score: 0,
          probabilities: { "0": 1 },
          legend: { "0": { description: "Low severity" } },
        },
      },
      usage: null,
    };
    render(<ResponseView response={response} isLoading={false} />);
    expect(screen.getByRole("meter", { name: '0: {"description":"Low severity"} probability' })).toHaveAttribute(
      "aria-valuenow",
      "100",
    );
    expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument();
  });

  it("shows an inline error message", () => {
    render(<ResponseView error="Upstream request failed" isLoading={false} />);

    expect(screen.getByRole("alert")).toHaveTextContent("Upstream request failed");
  });
});
