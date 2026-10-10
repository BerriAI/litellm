import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import ResponseView from "./ResponseView";
import type { SystemOneResponse } from "./lib/schemas";
import type { OpenAIDecisionsResponse } from "./lib/openAIDecisions";

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

describe("ResponseView integration", () => {
  it("shows OpenAI Decisions answer cards in Form view and the complete response in JSON view", async () => {
    const response: OpenAIDecisionsResponse = {
      model: "jev-latest",
      answers: [
        { name: null, type: "predicate", probability: 0 },
        {
          name: "route",
          type: "choice",
          choice: false,
          confidence: 0.8,
          probabilities: [
            { value: false, probability: 0.8 },
            { value: "other", probability: 0.2 },
          ],
        },
        {
          name: "severity",
          type: "score",
          score: 0,
          confidence: 0,
          probabilities: [
            { value: 0, label: "low", probability: 1 },
            { value: 1, label: "high", probability: 0 },
          ],
        },
        { name: "blocked", type: "refusal" },
        { name: "blocked", type: "refusal" },
      ],
      usage: { input_tokens: 3, output_tokens: 4, total_tokens: 7 },
      request_id: "keep-extra-fields",
    };
    const { rerender } = render(<ResponseView response={response} isLoading={false} view="form" />);
    expect(screen.getByText("Question 1")).toBeInTheDocument();
    expect(screen.getByText("0% yes")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "false probability" })).toHaveAttribute("aria-valuetext", "80%, selected");
    expect(screen.getByRole("meter", { name: "0: low probability" })).toHaveAttribute(
      "aria-valuetext",
      "100%, selected",
    );
    expect(screen.getByText("0% confidence")).toBeInTheDocument();
    expect(screen.getAllByText("Model refused to answer")).toHaveLength(2);
    expect(screen.queryByRole("region", { name: "Decisions response JSON" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Raw response" }));
    expect(screen.getByRole("region", { name: "Decisions response JSON" })).toHaveTextContent("keep-extra-fields");
    rerender(<ResponseView response={response} isLoading={false} view="json" />);
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByRole("region", { name: "Decisions response JSON" }).textContent ?? "")).toEqual(
      response,
    );
    rerender(<ResponseView response={response} isLoading={false} view="form" />);
    expect(screen.getByRole("meter", { name: "false probability" })).toHaveAttribute("aria-valuenow", "80");
  });

  it("shows System One raw response instead of cards in JSON view", () => {
    render(<ResponseView response={sampleResponse} isLoading={false} view="json" />);
    expect(screen.queryByRole("meter")).not.toBeInTheDocument();
    expect(JSON.parse(screen.getByRole("region", { name: "Decisions response JSON" }).textContent ?? "")).toEqual(
      sampleResponse,
    );
  });

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

  it("rounds the calibrated score pill to two decimals and leaves integers bare", () => {
    const precise: SystemOneResponse = {
      answers: {
        severity: { type: "score", score: 3.9809950180008897, probabilities: { "3": 0.02, "4": 0.98 } },
        clarity: { type: "score", score: 2, probabilities: { "2": 1 }, legend: { "2": "Clear enough" } },
      },
    };
    render(<ResponseView response={precise} isLoading={false} />);

    expect(screen.getByText("3.98")).toBeInTheDocument();
    expect(screen.queryByText("3.9809950180008897")).not.toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
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
