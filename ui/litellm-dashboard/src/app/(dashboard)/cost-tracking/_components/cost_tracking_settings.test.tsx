import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CostTrackingSettings from "./cost_tracking_settings";

// Mock sub-hooks so we can control their state without network calls
const mockDiscountConfig = vi.fn(() => ({}));
const mockMarginConfig = vi.fn(() => ({}));
const mockRemoveDiscount = vi.fn();
const mockRemoveMargin = vi.fn();

const stableDiscountCallbacks = {
  fetchDiscountConfig: vi.fn().mockResolvedValue(undefined),
  handleAddProvider: vi.fn().mockResolvedValue(true),
  handleRemoveProvider: mockRemoveDiscount,
  handleDiscountChange: vi.fn().mockResolvedValue(undefined),
};

const stableMarginCallbacks = {
  fetchMarginConfig: vi.fn().mockResolvedValue(undefined),
  handleAddMargin: vi.fn().mockResolvedValue(true),
  handleRemoveMargin: mockRemoveMargin,
  handleMarginChange: vi.fn().mockResolvedValue(undefined),
};

vi.mock("./use_discount_config", () => ({
  useDiscountConfig: () => ({ discountConfig: mockDiscountConfig(), ...stableDiscountCallbacks }),
}));

vi.mock("./use_margin_config", () => ({
  useMarginConfig: () => ({ marginConfig: mockMarginConfig(), ...stableMarginCallbacks }),
}));

vi.mock("./pricing_calculator/index", () => ({
  default: () => <div data-testid="pricing-calculator">Pricing Calculator</div>,
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/HelpLink", () => ({
  DocsMenu: () => null,
}));

vi.mock("./how_it_works", () => ({
  default: () => <div data-testid="how-it-works">How It Works</div>,
}));

vi.mock("@/components/provider_info_helpers", () => ({
  Providers: { OpenAI: "OpenAI" },
  provider_map: { OpenAI: "openai" },
  providerLogoMap: {},
  getProviderLogoAndName: (providerValue: string) => ({ logo: "", displayName: providerValue }),
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
    expect(container).toBeEmptyDOMElement();
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
    it("should open the Add Provider Discount modal when the button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      // The button lives inside the Provider Discounts accordion — click the header to expand first
      const accordionHeader = screen.getByText("Provider Discounts").closest("button");
      if (accordionHeader) {
        await user.click(accordionHeader);
      }

      const addButton = await screen.findByRole("button", { name: /add provider discount/i });
      await user.click(addButton);

      expect(await screen.findByRole("dialog", { name: "Add Provider Discount" })).toBeInTheDocument();
    });
  });

  describe("Add Provider Margin modal", () => {
    it("should open the Add Provider Margin modal when the button is clicked", async () => {
      const user = userEvent.setup();
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      const accordionHeader = screen.getByText("Fee/Price Margin").closest("button");
      if (accordionHeader) {
        await user.click(accordionHeader);
      }

      const addButton = await screen.findByRole("button", { name: /add provider margin/i });
      await user.click(addButton);

      expect(await screen.findByRole("dialog", { name: "Add Provider Margin" })).toBeInTheDocument();
    });
  });

  describe("removing a configured provider", () => {
    const expandAndRemove = async (section: string, actionName: string) => {
      const user = userEvent.setup();
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      await user.click(screen.getByText(section).closest("button")!);
      await user.click(await screen.findByRole("button", { name: actionName }));

      return user;
    };

    it("should ask to confirm before removing a discount", async () => {
      mockDiscountConfig.mockReturnValue({ openai: 0.05 });

      await expandAndRemove("Provider Discounts", "Remove discount for openai");

      expect(await screen.findByRole("button", { name: "Remove" })).toBeInTheDocument();
      expect(screen.getByText(/are you sure you want to remove the discount for openai\?/i)).toBeInTheDocument();
      expect(mockRemoveDiscount).not.toHaveBeenCalled();
    });

    it("should remove the discount once removal is confirmed", async () => {
      mockDiscountConfig.mockReturnValue({ openai: 0.05 });

      const user = await expandAndRemove("Provider Discounts", "Remove discount for openai");
      await user.click(await screen.findByRole("button", { name: "Remove" }));

      expect(mockRemoveDiscount).toHaveBeenCalledWith("openai");
    });

    it("should leave the discount in place when the confirmation is cancelled", async () => {
      mockDiscountConfig.mockReturnValue({ openai: 0.05 });

      const user = await expandAndRemove("Provider Discounts", "Remove discount for openai");
      await user.click(await screen.findByRole("button", { name: "Cancel" }));

      expect(mockRemoveDiscount).not.toHaveBeenCalled();
      expect(screen.queryByRole("button", { name: "Remove" })).not.toBeInTheDocument();
    });

    it("should hold the confirmation open while the removal is still in flight", async () => {
      mockDiscountConfig.mockReturnValue({ openai: 0.05 });
      const { promise, resolve: settleRemoval } = Promise.withResolvers<void>();
      mockRemoveDiscount.mockReturnValue(promise);

      const user = await expandAndRemove("Provider Discounts", "Remove discount for openai");
      await user.click(await screen.findByRole("button", { name: "Remove" }));

      const removing = await screen.findByRole("button", { name: "Removing…" });
      expect(removing).toBeDisabled();
      expect(screen.getByRole("button", { name: "Cancel" })).toBeDisabled();

      await act(async () => {
        settleRemoval();
      });

      await waitFor(() => {
        expect(screen.queryByRole("alertdialog")).not.toBeInTheDocument();
      });
      expect(mockRemoveDiscount).toHaveBeenCalledWith("openai");
    });

    it("should remove the margin once removal is confirmed", async () => {
      mockMarginConfig.mockReturnValue({ openai: 0.1 });

      const user = await expandAndRemove("Fee/Price Margin", "Remove margin for openai");
      expect(screen.getByText(/are you sure you want to remove the margin for openai\?/i)).toBeInTheDocument();
      await user.click(await screen.findByRole("button", { name: "Remove" }));

      expect(mockRemoveMargin).toHaveBeenCalledWith("openai");
    });
  });

  describe("empty state messages", () => {
    it("should show the empty state message when no discount config is loaded", async () => {
      mockDiscountConfig.mockReturnValue({});
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      const accordionHeader = screen.getByText("Provider Discounts").closest("button");
      if (accordionHeader) {
        await userEvent.setup().click(accordionHeader);
      }

      expect(await screen.findByText(/no provider discounts configured/i)).toBeInTheDocument();
    });

    it("should show the empty state message when no margin config is loaded", async () => {
      mockMarginConfig.mockReturnValue({});
      renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);

      const accordionHeader = screen.getByText("Fee/Price Margin").closest("button");
      if (accordionHeader) {
        await userEvent.setup().click(accordionHeader);
      }

      expect(await screen.findByText(/no provider margins configured/i)).toBeInTheDocument();
    });
  });
});
