import { Form } from "antd";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { describe, expect, it, vi } from "vitest";
import AddAutoRouterTab from "./add_auto_router_tab";
import { modelAvailableCall, testConnectionRequest } from "../networking";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";

vi.mock("../networking", () => ({
  modelAvailableCall: vi.fn(),
  testConnectionRequest: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn(),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: {
    fromBackend: vi.fn(),
    success: vi.fn(),
  },
}));

vi.mock("./ComplexityRouterConfig", () => ({
  default: ({ onChange }: { onChange: (tiers: Record<string, string>) => void }) => (
    <button
      type="button"
      onClick={() =>
        onChange({
          SIMPLE: "gpt-3.5-turbo",
          MEDIUM: "gpt-4o-mini",
          COMPLEX: "",
          REASONING: "",
        })
      }
    >
      Select complexity tiers
    </button>
  ),
}));

const renderAddAutoRouterTab = () => {
  const Component = () => {
    const [form] = Form.useForm();
    return <AddAutoRouterTab form={form} handleOk={vi.fn()} accessToken="test-token" userRole="proxy_admin" />;
  };

  return renderWithProviders(<Component />);
};

describe("AddAutoRouterTab", () => {
  it("tests complexity router connections with auto-router params", async () => {
    vi.mocked(modelAvailableCall).mockResolvedValue({ data: [] });
    vi.mocked(fetchAvailableModels).mockResolvedValue([
      { model_group: "gpt-3.5-turbo" },
      { model_group: "gpt-4o-mini" },
    ]);
    vi.mocked(testConnectionRequest).mockResolvedValue({ status: "success" });

    const user = userEvent.setup();
    renderAddAutoRouterTab();

    await user.type(screen.getByPlaceholderText("e.g., smart_router, auto_router_1"), "smart_router");
    await user.click(screen.getByRole("button", { name: "Select complexity tiers" }));
    await user.click(screen.getByRole("button", { name: "Test Connection" }));

    await waitFor(() =>
      expect(testConnectionRequest).toHaveBeenCalledWith(
        "test-token",
        {
          model: "auto_router/complexity_router",
          complexity_router_config: {
            tiers: {
              SIMPLE: "gpt-3.5-turbo",
              MEDIUM: "gpt-4o-mini",
              COMPLEX: "",
              REASONING: "",
            },
          },
          complexity_router_default_model: "gpt-4o-mini",
        },
        {},
        undefined,
      ),
    );
  });
});
