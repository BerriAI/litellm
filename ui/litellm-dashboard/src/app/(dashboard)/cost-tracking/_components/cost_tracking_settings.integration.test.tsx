import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../tests/test-utils";
import CostTrackingSettings from "./cost_tracking_settings";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";

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
const stableUnpricedCallbacks = {
  fetchBlockUnpriced: vi.fn().mockResolvedValue(undefined),
  setBlockUnpriced: vi.fn().mockResolvedValue(undefined),
};

vi.mock("./use_block_unpriced_config", () => ({
  useBlockUnpricedConfig: () => ({ blockUnpriced: false, isUpdating: false, ...stableUnpricedCallbacks }),
}));

vi.mock("./use_discount_config", () => ({
  useDiscountConfig: () => {
    const { t } = useLanguage();
    // 模拟真实 hook：错误提示使用 t，语言变化时回调引用也变化。
    const fetchDiscountConfig = React.useMemo(() => {
      void t;
      return () => stableDiscountCallbacks.fetchDiscountConfig();
    }, [t]);
    return { discountConfig: mockDiscountConfig(), ...stableDiscountCallbacks, fetchDiscountConfig };
  },
}));

vi.mock("./use_margin_config", () => ({
  useMarginConfig: () => {
    const { t } = useLanguage();
    const fetchMarginConfig = React.useMemo(() => {
      void t;
      return () => stableMarginCallbacks.fetchMarginConfig();
    }, [t]);
    return { marginConfig: mockMarginConfig(), ...stableMarginCallbacks, fetchMarginConfig };
  },
}));

vi.mock("./pricing_calculator/index", () => ({
  default: () => <div data-testid="pricing-calculator">Pricing Calculator</div>,
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/DocsMenu", () => ({
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

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("zh-CN")}>切换语言</button>;
}
describe("CostTrackingSettings submit paths", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
    mockDiscountConfig.mockReturnValue({});
    mockMarginConfig.mockReturnValue({});
  });

  it("语言切换不重新加载折扣、加价或可用模型", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="en" userId="pricing-language-test" storageScope="pricing-language-test">
        <LanguageSwitch />
        <CostTrackingSettings {...ADMIN_PROPS} />
      </LanguageProvider>,
    );
    await waitFor(() => expect(stableDiscountCallbacks.fetchDiscountConfig).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "切换语言" }));
    expect(screen.getByText("费用跟踪设置")).toBeInTheDocument();
    expect(stableDiscountCallbacks.fetchDiscountConfig).toHaveBeenCalledTimes(1);
    expect(stableMarginCallbacks.fetchMarginConfig).toHaveBeenCalledTimes(1);
  });

  const openDiscountModal = async (user: ReturnType<typeof userEvent.setup>) => {
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    const header = screen.getByText("Provider Discounts").closest("button");
    if (header) await user.click(header);
    await user.click(await screen.findByRole("button", { name: /add provider discount/i }));
    await screen.findByRole("dialog", { name: "Add Provider Discount" });
  };

  const submitDiscount = () =>
    screen
      .getAllByRole("button")
      .filter((button) => (button.textContent || "").trim() === "Add Provider Discount")
      .pop()!;

  it("requests the discount exactly once per click", async () => {
    const user = userEvent.setup();
    await openDiscountModal(user);

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click((await screen.findAllByRole("option"))[0]);
    fireEvent.change(screen.getByLabelText(/Discount Percentage/i), { target: { value: "5" } });
    await user.click(submitDiscount());

    await waitFor(() => expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalled());
    expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalledTimes(1);
    expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalledWith("OpenAI", "5", "");
  });

  it("passes the model pattern through to the add handler", async () => {
    const user = userEvent.setup();
    await openDiscountModal(user);

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click((await screen.findAllByRole("option"))[0]);
    fireEvent.change(screen.getByLabelText(/Model pattern/i), { target: { value: "claude-*" } });
    fireEvent.change(screen.getByLabelText(/Discount Percentage/i), { target: { value: "20" } });
    await user.click(submitDiscount());

    await waitFor(() => expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalled());
    expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalledWith("OpenAI", "20", "claude-*");
  });

  it("requests the margin exactly once per click", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    const header = screen.getByText("Fee/Price Margin").closest("button");
    if (header) await user.click(header);
    await user.click(await screen.findByRole("button", { name: /add provider margin/i }));
    await screen.findByRole("dialog", { name: "Add Provider Margin" });

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click((await screen.findAllByRole("option"))[0]);
    fireEvent.change(screen.getByLabelText(/Margin Percentage/i), { target: { value: "10" } });

    const submit = screen
      .getAllByRole("button")
      .filter((button) => (button.textContent || "").trim() === "Add Provider Margin")
      .pop()!;
    await user.click(submit);

    await waitFor(() => expect(stableMarginCallbacks.handleAddMargin).toHaveBeenCalled());
    expect(stableMarginCallbacks.handleAddMargin).toHaveBeenCalledTimes(1);
  });

  it("requests the discount exactly once when Enter is pressed in the discount field", async () => {
    const user = userEvent.setup();
    await openDiscountModal(user);

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click((await screen.findAllByRole("option"))[0]);
    await user.type(screen.getByLabelText(/Discount Percentage/i), "5{Enter}");

    await waitFor(() => expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalled());
    expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalledTimes(1);
    expect(stableDiscountCallbacks.handleAddProvider).toHaveBeenCalledWith("OpenAI", "5", "");
  });

  it("requests the margin exactly once when Enter is pressed in the percentage field", async () => {
    const user = userEvent.setup();
    renderWithProviders(<CostTrackingSettings {...ADMIN_PROPS} />);
    const header = screen.getByText("Fee/Price Margin").closest("button");
    if (header) await user.click(header);
    await user.click(await screen.findByRole("button", { name: /add provider margin/i }));
    await screen.findByRole("dialog", { name: "Add Provider Margin" });

    await user.click(screen.getAllByRole("combobox")[0]);
    await user.click((await screen.findAllByRole("option"))[0]);
    await user.type(screen.getByLabelText(/Margin Percentage/i), "10{Enter}");

    await waitFor(() => expect(stableMarginCallbacks.handleAddMargin).toHaveBeenCalled());
    expect(stableMarginCallbacks.handleAddMargin).toHaveBeenCalledTimes(1);
  });
});
