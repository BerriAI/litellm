import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import i18n from "@/i18n/i18n";
import type { OverviewTotals } from "./overviewData";
import UsageStatStrip from "./UsageStatStrip";

const totals: OverviewTotals = {
  spend: 42.5,
  requests: 10,
  successful: 8,
  failed: 2,
  tokens: 9123456,
  inputTokens: 8123456,
  outputTokens: 1000000,
  cacheReadTokens: 5123456,
  cacheWriteTokens: 2000000,
  avgCostPerRequest: 4.25,
  avgLatencyMs: 250,
  successRate: 80,
};

const asRendered = (text: string) => text.replace(/\s+/g, " ");
const SECONDS = {
  style: "unit",
  unit: "second",
  unitDisplay: "narrow",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
} as const;

const defaultProps = {
  results: [],
  totals,
  loading: false,
  requestCountsPending: false,
  budget: null,
};

describe("UsageStatStrip token breakdown", () => {
  it("should render the existing totals and open the full token counts", async () => {
    const user = userEvent.setup();
    render(<UsageStatStrip {...defaultProps} />);

    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("80.0% success rate")).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "View token breakdown" }));
    const breakdown = await screen.findByRole("dialog", { name: "What counts toward total tokens?" });
    expect(breakdown).toHaveTextContent(/Input tokens\s*8,123,456/);
    expect(breakdown).toHaveTextContent(/Cache reads\s*5,123,456/);
    expect(breakdown).toHaveTextContent(/Cache writes\s*2,000,000/);
    expect(breakdown).toHaveTextContent(/Output tokens\s*1,000,000/);
    expect(breakdown).toHaveTextContent(/Total tokens\s*9,123,456/);
    expect(breakdown).toHaveTextContent("Input includes cache reads and writes.");
    expect(breakdown).toHaveTextContent("Compare the same person, client and time range");
  });

  it("should hide the token breakdown and request counts while usage is loading", () => {
    render(<UsageStatStrip {...defaultProps} loading requestCountsPending />);

    expect(screen.getByText("Total Tokens")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "View token breakdown" })).not.toBeInTheDocument();
    expect(screen.queryByText("ok")).not.toBeInTheDocument();
    expect(screen.queryByText("failed")).not.toBeInTheDocument();
  });
});

describe("UsageStatStrip in another UI language", () => {
  afterEach(async () => {
    await i18n.changeLanguage("en");
    i18n.removeResourceBundle("de", "translation");
  });

  it("should format every value in the selected language and keep money in US dollars", async () => {
    i18n.addResourceBundle("de", "translation", {});
    await i18n.changeLanguage("de");
    const german = new Intl.NumberFormat("de");
    const dollars = new Intl.NumberFormat("de", { style: "currency", currency: "USD" });
    const euros = new Intl.NumberFormat("de", { style: "currency", currency: "EUR" });
    const dollarsToFourPlaces = new Intl.NumberFormat("de", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 4,
    });
    const oneDecimal = new Intl.NumberFormat("de", { minimumFractionDigits: 1 });
    const milliseconds = new Intl.NumberFormat("de", { style: "unit", unit: "millisecond", unitDisplay: "narrow" });

    render(<UsageStatStrip {...defaultProps} budget={1500} />);

    expect(screen.getByText(asRendered(dollars.format(42.5)))).toBeInTheDocument();
    expect(screen.getByText(asRendered(`of ${dollars.format(1500)} budget`))).toBeInTheDocument();
    expect(screen.getByText(asRendered(dollarsToFourPlaces.format(4.25)))).toBeInTheDocument();
    expect(screen.getByText(asRendered(milliseconds.format(250)))).toBeInTheDocument();
    expect(screen.getByText(`${oneDecimal.format(80)}% success rate`)).toBeInTheDocument();
    expect(screen.getByTitle(`${german.format(9123456)} tokens`)).toBeInTheDocument();
    expect(screen.queryByText(asRendered(euros.format(42.5)))).not.toBeInTheDocument();
  });

  it("should format large counts, compact tokens, seconds and the token breakdown in the selected language", async () => {
    const user = userEvent.setup();
    i18n.addResourceBundle("de", "translation", {});
    await i18n.changeLanguage("de");
    const german = new Intl.NumberFormat("de");
    const compact = new Intl.NumberFormat("de", { notation: "compact", maximumFractionDigits: 1 });
    const large = { ...totals, requests: 8123456, successful: 7654321, failed: 469135, avgLatencyMs: 1234 };

    render(<UsageStatStrip {...defaultProps} totals={large} />);

    expect(screen.getByText(german.format(8123456))).toBeInTheDocument();
    expect(screen.getByText(german.format(7654321))).toBeInTheDocument();
    expect(screen.getByText(german.format(469135))).toBeInTheDocument();
    expect(screen.getByText(asRendered(compact.format(9123456)))).toBeInTheDocument();
    expect(screen.getByText(asRendered(new Intl.NumberFormat("de", SECONDS).format(1.234)))).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "View token breakdown" }));
    const breakdown = await screen.findByRole("dialog", { name: "What counts toward total tokens?" });
    [8123456, 5123456, 2000000, 1000000, 9123456].forEach((count) =>
      expect(breakdown).toHaveTextContent(german.format(count)),
    );
  });
});
