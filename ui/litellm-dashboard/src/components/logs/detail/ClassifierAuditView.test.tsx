import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ClassifierAuditView } from "./ClassifierAuditView";

vi.mock("./JsonViewer", () => ({
  JsonViewer: ({ data }: { data: unknown }) => <pre>{JSON.stringify(data)}</pre>,
}));

describe("ClassifierAuditView", () => {
  it("separates and copies the provider input, source request, and returned verdict", async () => {
    const user = userEvent.setup();
    const input = { system: "classification rubric", messages: [{ role: "user", content: "classify this" }] };
    render(
      <ClassifierAuditView
        request={{ classifier_input: input, originating_request_masked: { input: "source-only", api_key: "REDACTED" } }}
        response={{ tier: "SIMPLE", reason: "a greeting" }}
      />,
    );
    const classifier = within(screen.getByRole("region", { name: "Classifier input" }));
    expect(classifier.getByText(/classification rubric/)).toBeInTheDocument();
    expect(classifier.queryByText(/source-only/)).not.toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: "Originating request, credentials masked" })).getByText(/source-only/),
    ).toBeInTheDocument();
    expect(
      within(screen.getByRole("region", { name: "Classifier response" })).getByText(/a greeting/),
    ).toBeInTheDocument();
    await user.click(classifier.getByRole("button", { name: "Copy Classifier input" }));
    expect(await navigator.clipboard.readText()).toBe(JSON.stringify(input, null, 2));
  });

  it("does not present legacy source messages as captured classifier input", () => {
    render(<ClassifierAuditView request={{ messages: [{ content: "legacy source" }] }} response={undefined} />);
    expect(screen.getAllByText("Not captured or message logging disabled")).toHaveLength(3);
    expect(screen.queryByRole("button", { name: "Copy Classifier input" })).not.toBeInTheDocument();
  });

  it("labels truncated input without marking a complete source request as truncated", () => {
    render(
      <ClassifierAuditView
        request={{
          classifier_input: { system: "partial rubric...litellm_truncated" },
          originating_request_masked: { input: "source" },
        }}
        response={{ tier: "SIMPLE" }}
      />,
    );
    expect(within(screen.getByRole("region", { name: "Classifier input" })).getByRole("status")).toHaveTextContent(
      "This stored copy is truncated",
    );
    expect(
      within(screen.getByRole("region", { name: "Originating request, credentials masked" })).queryByRole("status"),
    ).not.toBeInTheDocument();
  });
});
