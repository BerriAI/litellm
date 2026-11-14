import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "@/components/chat/types";
import { toast } from "@/lib/toast";
import { AriaAnswer } from "./AriaAnswer";
import { createAriaAnswerDocument, exportAriaAnswer } from "./answerExport";
import type { UsageChart } from "./usageChart";

vi.mock("./answerExport", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./answerExport")>()),
  exportAriaAnswer: vi.fn(() => true),
}));

const message: ChatMessage = {
  id: "answer-1",
  role: "assistant",
  content: "## Spend overview\n\nThe **primary model** accounts for most of your spend.",
  timestamp: 1,
};
const chart: UsageChart = {
  kind: "bar",
  title: "Spend by model",
  period: "2026-10-19 to 2026-10-23 (UTC)",
  scope: "Your usage",
  unit: "usd",
  total: 4,
  data: [
    { label: "Primary", value: 3.25 },
    { label: "Secondary", value: 0.75 },
  ],
};

beforeEach(() => {
  vi.mocked(exportAriaAnswer).mockReset().mockReturnValue(true);
  vi.mocked(toast.error).mockClear();
});

describe("AriaAnswer", () => {
  it("should render a text-only Markdown answer with one export action", () => {
    render(<AriaAnswer message={message} />);

    expect(screen.getByRole("article", { name: "Aria Agent answer" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Spend overview", level: 2 })).toBeInTheDocument();
    expect(screen.getByText("primary model")).toBeInTheDocument();
    expect(screen.queryByRole("figure")).not.toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Export answer" })).toHaveLength(1);
    expect(screen.queryByRole("button", { name: /copy/i })).not.toBeInTheDocument();
  });

  it("should present the chart's exact accessible values alongside the formatted explanation", () => {
    render(<AriaAnswer message={message} question="Which model costs the most?" chart={chart} />);

    const figure = screen.getByRole("figure", { name: "Spend by model" });
    const values = within(figure).getByRole("table", { name: "Spend by model values" });
    expect(within(values).getByRole("row", { name: "Primary $3.25" })).toBeInTheDocument();
    expect(within(values).getByRole("row", { name: "Secondary $0.75" })).toBeInTheDocument();
    expect(within(figure).getByText("$4.00")).toBeInTheDocument();
    expect(within(figure).getByText("Your usage · 2026-10-19 to 2026-10-23 (UTC)")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Spend overview" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /copy/i })).not.toBeInTheDocument();
  });

  it("should export the current question, explanation and chart without including adjacent receipts", () => {
    const question = "Which model costs the most?";
    render(
      <>
        <aside>Private action receipt</aside>
        <AriaAnswer message={message} question={question} chart={chart} />
      </>,
    );
    act(() => fireEvent.click(screen.getByRole("button", { name: "Export answer" })));

    expect(exportAriaAnswer).toHaveBeenCalledOnce();
    const [exportedQuestion, exportedAnswer] = vi.mocked(exportAriaAnswer).mock.calls[0];
    expect(exportedQuestion).toBe(question);
    expect(within(exportedAnswer).getByRole("figure", { name: "Spend by model" })).toBeInTheDocument();
    expect(within(exportedAnswer).getByRole("heading", { name: "Spend overview" })).toBeInTheDocument();
    expect(within(exportedAnswer).queryByText("Private action receipt")).not.toBeInTheDocument();
    expect(toast.error).not.toHaveBeenCalled();
  });

  it("should explain when the PDF print dialog cannot open", () => {
    vi.mocked(exportAriaAnswer).mockReturnValue(false);
    render(<AriaAnswer message={message} />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Export answer" })));

    expect(exportAriaAnswer).toHaveBeenCalledWith("", expect.any(HTMLElement));
    expect(toast.error).toHaveBeenCalledWith("Could not open the PDF print dialog. Please try again.");
  });

  it("should preserve full chart labels and exact values in the exported answer", () => {
    const question = "Which deployment costs the most?";
    const detailedChart: UsageChart = {
      ...chart,
      total: null,
      data: [
        { label: "provider/long-deployment-primary", value: 0.000123 },
        { label: "provider/long-deployment-secondary", value: null },
        { label: "Unused deployment", value: 0 },
      ],
    };
    render(<AriaAnswer message={message} question={question} chart={detailedChart} />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Export answer" })));
    const [exportedQuestion, exportedAnswer] = vi.mocked(exportAriaAnswer).mock.calls[0];

    const report = createAriaAnswerDocument(exportedQuestion, exportedAnswer);
    render(<iframe title="Exported answer preview" />);
    const preview = screen.getByTitle<HTMLIFrameElement>("Exported answer preview").contentDocument!;
    preview.replaceChild(preview.importNode(report.documentElement, true), preview.documentElement);

    const table = within(preview.body).getByRole("table");
    expect(within(table).getByRole("row", { name: "provider/long-deployment-primary $0.000123" })).toBeInTheDocument();
    expect(
      within(table).getByRole("row", { name: "provider/long-deployment-secondary Unavailable" }),
    ).toBeInTheDocument();
    expect(within(table).getByRole("row", { name: "Unused deployment $0.00" })).toBeInTheDocument();
    expect(preview.body).toHaveTextContent(question);
    expect(within(preview.body).queryByRole("button")).not.toBeInTheDocument();
  });

  it("should distinguish missing token counts from zero in charts and their accessible data", () => {
    render(
      <AriaAnswer
        message={message}
        chart={{
          ...chart,
          title: "Token breakdown",
          unit: "tokens",
          total: 22,
          data: [
            { label: "Prompt", value: 14 },
            { label: "Completion", value: 8 },
            { label: "Cache read", value: null },
            { label: "Cache creation", value: 0 },
          ],
          note: "Cache counts are reported separately.",
        }}
      />,
    );

    const values = screen.getByRole("table", { name: "Token breakdown values" });
    expect(within(values).getByRole("row", { name: "Cache read Unavailable" })).toBeInTheDocument();
    expect(within(values).getByRole("row", { name: "Cache creation 0" })).toBeInTheDocument();
    expect(screen.getByText(/Values unavailable for: Cache read\./)).toBeInTheDocument();
  });
});
