import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { screen } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import CostTrackingSettings from "./cost_tracking_settings";

// Mock sub-hooks so we can control their state without network calls
const mockDiscountConfig = vi.fn(() => ({}));
const mockMarginConfig = vi.fn(() => ({}));

vi.mock("./use_discount_config", () => ({
  useDiscountConfig: () => ({
    discountConfig: mockDiscountConfig(),
    fetchDiscountConfig: vi.fn().mockResolvedValue(undefined),
    handleAddProvider: vi.fn().mockResolvedValue(true),
    handleRemoveProvider: vi.fn().mockResolvedValue(undefined),
    handleDiscountChange: vi.fn().mockResolvedValue(undefined),
  }),
}));

vi.mock("./use_margin_config", () => ({
  useMarginConfig: () => ({
    marginConfig: mockMarginConfig(),
    fetchMarginConfig: vi.fn().mockResolvedValue(undefined),
    handleAddMargin: vi.fn().mockResolvedValue(true),
    handleRemoveMargin: vi.fn().mockResolvedValue(undefined),
    handleMarginChange: vi.fn().mockResolvedValue(undefined),
  }),
}));

vi.mock("./pricing_calculator/index", () => ({
  default: () => <div data-testid="pricing-calculator">Pricing Calculator</div>,
}));

vi.mock("../playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("../HelpLink", () => ({
  DocsMenu: () => null,
}));

vi.mock("./how_it_works", () => ({
  default: () => <div data-testid="how-it-works">How It Works</div>,
}));

vi.mock("../provider_info_helpers", () => ({
  Providers: { OpenAI: "OpenAI" },
  provider_map: { OpenAI: "openai" },
  providerLogoMap: {},
}));

vi.mock("./provider_display_helpers", () => ({
  getProviderDisplayInfo: vi.fn(() => ({ displayName: "OpenAI", logo: "", enumKey: "OpenAI" })),
  handleImageError: vi.fn(),
}));

const ADMIN_PROPS = {
  userID: "user-1",
  userRole: "proxy_admin",
  accessToken: "test-token",
};

describe("CostTrackingSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockDiscountConfig.mockReturnValue({});
    mockMarginConfig.mockReturnValue({});
  });

  it("should return nothing when accessToken is null", () => {
    const { container } = renderWithProviders(
      <CostTrackingSettings userID="user-1" userRole="proxy_admin" accessToken={null} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("should render the page title", () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    expect(screen.getByText("Cost Tracking Settings")).toBeInTheDocument();
  });

  it("should show the Provider Discounts accordion header for proxy_admin", () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    expect(screen.getByText("Provider Discounts")).toBeInTheDocument();
  });

  it("should show the Fee/Price Margin accordion header for proxy_admin", () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    expect(screen.getByText("Fee/Price Margin")).toBeInTheDocument();
  });

  it("should always show the Pricing Calculator section", () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    // The accordion header text appears in the DOM; getAllByText tolerates duplicates
    expect(screen.getAllByText("Pricing Calculator").length).toBeGreaterThan(0);
  });

  it("should show the pricing calculator component", async () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    expect(await screen.findByTestId("pricing-calculator")).toBeInTheDocument();
  });

  it("should not show Provider Discounts section for a non-admin role", () => {
    renderWithProviders(<CostTrackingSettings userID="user-1" userRole="internal_user" accessToken="test-token" />);
    expect(screen.queryByText("Provider Discounts")).not.toBeInTheDocument();
  });

  it("should not show Fee/Price Margin section for a non-admin role", () => {
    renderWithProviders(<CostTrackingSettings userID="user-1" userRole="internal_user" accessToken="test-token" />);
    expect(screen.queryByText("Fee/Price Margin")).not.toBeInTheDocument();
  });

  it("should show Provider Discounts for the 'Admin' role as well", () => {
    renderWithProviders(<CostTrackingSettings userID="user-1" userRole="Admin" accessToken="test-token" />);
    expect(screen.getByText("Provider Discounts")).toBeInTheDocument();
  });

  it("should show the subtitle describing discount/margin configuration", () => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    expect(screen.getByText(/configure cost discounts and margins/i)).toBeInTheDocument();
  });

  describe("Add Provider Discount modal", () => {
    it("should render an Add Provider Discount trigger button", () => {
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
      expect(screen.getByRole("button", { name: /add provider discount/i })).toBeInTheDocument();
    });
  });

  describe("Add Provider Margin modal", () => {
    it("should render an Add Provider Margin trigger button", () => {
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
      expect(screen.getByRole("button", { name: /add provider margin/i })).toBeInTheDocument();
    });
  });

  describe("empty state messages", () => {
    it("should show the empty state message when no discount config is loaded", async () => {
      mockDiscountConfig.mockReturnValue({});
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      expect(await screen.findByText(/no provider discounts configured/i)).toBeInTheDocument();
    });

    it("should show the empty state message when no margin config is loaded", async () => {
      mockMarginConfig.mockReturnValue({});
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      expect(await screen.findByText(/no provider margins configured/i)).toBeInTheDocument();
    });
  });
});
