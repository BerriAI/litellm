import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import SpendByProvider from "./SpendByProvider";

vi.mock("../../../shared/chart_loader", () => ({
  ChartLoader: ({ isDateChanging }: { isDateChanging: boolean }) => (
    <div data-testid="chart-loader">
      {isDateChanging ? "Processing date selection..." : "Loading chart data..."}
    </div>
  ),
}));

vi.mock("../../../molecules/models/ProviderLogo", () => ({
  ProviderLogo: ({ provider }: { provider: string }) => (
    <div data-testid={`provider-logo-${provider}`}>{provider}</div>
  ),
}));

describe("SpendByProvider", () => {
  const mockProviderSpend = [
    {
      provider: "openai",
      spend: 150.5,
      requests: 100,
      successful_requests: 95,
      failed_requests: 5,
      tokens: 50000,
    },
    {
      provider: "anthropic",
      spend: 200.75,
      requests: 120,
      successful_requests: 115,
      failed_requests: 5,
      tokens: 75000,
    },
    {
      provider: "unknown",
      spend: 0,
      requests: 10,
      successful_requests: 0,
      failed_requests: 10,
      tokens: 0,
    },
    {
      provider: "google",
      spend: 0,
      requests: 0,
      successful_requests: 0,
      failed_requests: 0,
      tokens: 0,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={[]} />);
    expect(screen.getByText("Spend by Provider")).toBeInTheDocument();
  });

  it("should display the title", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={[]} />);
    expect(screen.getByText("Spend by Provider")).toBeInTheDocument();
  });

  it("should display Show Zero Spend toggle", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={[]} />);
    expect(screen.getByText("Show Zero Spend")).toBeInTheDocument();
    expect(screen.getAllByRole("switch")[0]).toBeInTheDocument();
  });

  it("should display Show Unknown toggle", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={[]} />);
    expect(screen.getByText("Show Unknown")).toBeInTheDocument();
    expect(screen.getAllByRole("switch")[1]).toBeInTheDocument();
  });

  it("should display table headers", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
    expect(screen.getByText("Successful")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("Tokens")).toBeInTheDocument();
  });

  it("should display provider data in table", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getAllByText("openai").length).toBeGreaterThan(0);
    expect(screen.getAllByText("anthropic").length).toBeGreaterThan(0);
    expect(screen.getByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("$200.75")).toBeInTheDocument();
  });

  it("should display formatted spend values with two decimal places", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByText("$150.50")).toBeInTheDocument();
    expect(screen.getByText("$200.75")).toBeInTheDocument();
  });

  it("should display successful requests with locale formatting", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByText("95")).toBeInTheDocument();
    expect(screen.getByText("115")).toBeInTheDocument();
  });

  it("should display failed requests with locale formatting", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getAllByText("5").length).toBeGreaterThan(0);
  });

  it("should display tokens with locale formatting", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByText("50,000")).toBeInTheDocument();
    expect(screen.getByText("75,000")).toBeInTheDocument();
  });

  it("should display provider logos", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByTestId("provider-logo-openai")).toBeInTheDocument();
    expect(screen.getByTestId("provider-logo-anthropic")).toBeInTheDocument();
  });

  it("should filter out providers with zero spend by default", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getAllByText("openai").length).toBeGreaterThan(0);
    expect(screen.getAllByText("anthropic").length).toBeGreaterThan(0);
    expect(screen.queryByText("google")).not.toBeInTheDocument();
  });

  it("should filter out unknown provider by default", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.queryByText("unknown")).not.toBeInTheDocument();
  });

  it("should display ChartLoader when loading is true", () => {
    render(<SpendByProvider loading={true} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getByTestId("chart-loader")).toBeInTheDocument();
    expect(screen.getByText("Loading chart data...")).toBeInTheDocument();
  });

  it("should display ChartLoader with date changing message when isDateChanging is true", () => {
    render(<SpendByProvider loading={true} isDateChanging={true} providerSpend={mockProviderSpend} />);
    expect(screen.getByTestId("chart-loader")).toBeInTheDocument();
    expect(screen.getByText("Processing date selection...")).toBeInTheDocument();
  });

  it("should not display table when loading is true", () => {
    render(<SpendByProvider loading={true} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.queryByText("Provider")).not.toBeInTheDocument();
  });

  it("should handle empty provider spend array", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={[]} />);
    expect(screen.getByText("Provider")).toBeInTheDocument();
    expect(screen.getByText("Spend")).toBeInTheDocument();
  });

  it("should handle provider with null provider name", () => {
    const providerSpendWithNull = [
      {
        provider: null as unknown as string,
        spend: 100,
        requests: 50,
        successful_requests: 45,
        failed_requests: 5,
        tokens: 25000,
      },
    ];
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={providerSpendWithNull} />);
    expect(screen.getByText("$100.00")).toBeInTheDocument();
  });

  it("should handle provider with empty string provider name", () => {
    const providerSpendWithEmpty = [
      {
        provider: "",
        spend: 100,
        requests: 50,
        successful_requests: 45,
        failed_requests: 5,
        tokens: 25000,
      },
    ];
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={providerSpendWithEmpty} />);
    expect(screen.getByText("$100.00")).toBeInTheDocument();
  });

  it("should display large token numbers with comma formatting", () => {
    const providerSpendWithLargeTokens = [
      {
        provider: "openai",
        spend: 1000,
        requests: 1000,
        successful_requests: 950,
        failed_requests: 50,
        tokens: 1234567,
      },
    ];
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={providerSpendWithLargeTokens} />);
    expect(screen.getByText("1,234,567")).toBeInTheDocument();
  });

  it("should filter data correctly when both toggles are off", () => {
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={mockProviderSpend} />);
    expect(screen.getAllByText("openai").length).toBeGreaterThan(0);
    expect(screen.getAllByText("anthropic").length).toBeGreaterThan(0);
    expect(screen.queryByText("google")).not.toBeInTheDocument();
    expect(screen.queryByText("unknown")).not.toBeInTheDocument();
  });

  it("should include all providers with spend greater than zero by default", () => {
    const providerSpendWithMixed = [
      {
        provider: "provider1",
        spend: 0.01,
        requests: 10,
        successful_requests: 10,
        failed_requests: 0,
        tokens: 1000,
      },
      {
        provider: "provider2",
        spend: 0,
        requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        tokens: 0,
      },
    ];
    render(<SpendByProvider loading={false} isDateChanging={false} providerSpend={providerSpendWithMixed} />);
    expect(screen.getAllByText("provider1").length).toBeGreaterThan(0);
    expect(screen.queryByText("provider2")).not.toBeInTheDocument();
  });
});
