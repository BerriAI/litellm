import { screen, waitFor, renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ModelConnectionTest from "./model_connection_test";

const prepareModelAddRequest = vi.fn();
const testConnectionRequest = vi.fn();

vi.mock("./handle_add_model_submit", () => ({
  prepareModelAddRequest: (...args: unknown[]) => prepareModelAddRequest(...args),
}));

vi.mock("../networking", () => ({
  testConnectionRequest: (...args: unknown[]) => testConnectionRequest(...args),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

describe("ModelConnectionTest", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a friendly error instead of crashing when no deployments are prepared", async () => {
    // Auto Router forms have no model_mappings, so prepareModelAddRequest resolves to []
    prepareModelAddRequest.mockResolvedValue([]);

    renderWithProviders(
      <ModelConnectionTest formValues={{ auto_router_name: "smart_router" }} accessToken="token" testMode="chat" />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("connection-failure-msg")).toBeInTheDocument();
    });

    expect(screen.getByText("Failed to prepare model data. Please check your form inputs.")).toBeInTheDocument();
    expect(screen.queryByText(/Cannot destructure/i)).not.toBeInTheDocument();
    expect(testConnectionRequest).not.toHaveBeenCalled();
  });

  it("tests the injected deployment instead of the default add-model path", async () => {
    testConnectionRequest.mockResolvedValue({ status: "success" });
    const prepareDeployments = vi
      .fn()
      .mockResolvedValue([{ litellmParamsObj: { model: "gpt-4o-mini" }, modelInfoObj: {}, modelName: "gpt-4o-mini" }]);

    renderWithProviders(
      <ModelConnectionTest
        formValues={{ auto_router_name: "smart_router" }}
        accessToken="token"
        testMode="chat"
        prepareDeployments={prepareDeployments}
      />,
    );

    await waitFor(() => {
      expect(screen.getByTestId("connection-success-msg")).toBeInTheDocument();
    });

    expect(prepareModelAddRequest).not.toHaveBeenCalled();
    expect(testConnectionRequest).toHaveBeenCalledWith("token", { model: "gpt-4o-mini" }, {}, "chat");
  });
});
