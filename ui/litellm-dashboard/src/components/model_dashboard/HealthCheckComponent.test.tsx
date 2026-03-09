/* @vitest-environment jsdom */
import { act, render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import HealthCheckComponent from "./HealthCheckComponent";

const mockIndividualModelHealthCheckCall = vi.fn();
const mockLatestHealthChecksCall = vi.fn();

vi.mock("../networking", () => ({
  individualModelHealthCheckCall: (...args: unknown[]) => mockIndividualModelHealthCheckCall(...args),
  latestHealthChecksCall: (...args: unknown[]) => mockLatestHealthChecksCall(...args),
}));

describe("HealthCheckComponent", () => {
  const getDisplayModelName = (model: { model_name?: string }) => model.model_name ?? "";

  beforeEach(() => {
    vi.clearAllMocks();
    mockLatestHealthChecksCall.mockResolvedValue({ latest_health_checks: {} });
    mockIndividualModelHealthCheckCall.mockResolvedValue({
      healthy_count: 1,
      unhealthy_count: 0,
      healthy_endpoints: [],
      unhealthy_endpoints: [],
    });
  });

  it("should render the health check section", async () => {
    const modelData = {
      data: [
        {
          model_name: "gpt-4",
          model_info: { id: "deployment-1" },
          litellm_model_name: "gpt-4",
        },
      ],
    };

    await act(async () => {
      render(
        <HealthCheckComponent
          accessToken="token"
          modelData={modelData}
          all_models_on_proxy={["deployment-1"]}
          getDisplayModelName={getDisplayModelName}
        />,
      );
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });

    expect(screen.getByText("Model Health Status")).toBeInTheDocument();
    expect(
      screen.getByText("Run health checks on individual models to verify they are working correctly"),
    ).toBeInTheDocument();
  });

  it("should call individualModelHealthCheckCall with model id when run health check is triggered", async () => {
    const modelData = {
      data: [
        {
          model_name: "gpt-4",
          model_info: { id: "deployment-abc-123" },
          litellm_model_name: "gpt-4",
        },
      ],
    };

    render(
      <HealthCheckComponent
        accessToken="token-123"
        modelData={modelData}
        all_models_on_proxy={["deployment-abc-123"]}
        getDisplayModelName={getDisplayModelName}
      />,
    );

    const runButtons = screen.getAllByTestId("run-health-check-btn");
    expect(runButtons.length).toBeGreaterThanOrEqual(1);
    const runButton = runButtons[0];

    await act(async () => {
      runButton.click();
    });

    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token-123", "deployment-abc-123");
    expect(mockIndividualModelHealthCheckCall).not.toHaveBeenCalledWith("token-123", "gpt-4");
  });

  describe("latest_health_checks keyed by model id", () => {
    it("should show status from latest_health_checks when keys match model ids", async () => {
      const modelData = {
        data: [
          { 
            model_name: "gpt-4", 
            model_info: { id: "id-alpha" }, 
            litellm_model_name: "gpt-4",  
          },
          { 
            model_name: "gpt-4", 
            model_info: { id: "id-beta" }, 
            litellm_model_name: "gpt-4",
          },
        ],
      };

      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "id-alpha": {
            status: "healthy",
            checked_at: "2024-01-15T10:00:00Z",
            error_message: null,
          },
          "id-beta": {
            status: "unhealthy",
            checked_at: "2024-01-15T10:05:00Z",
            error_message: "Connection failed",
          },
        },
      });

      await act(async () => {
        render(
          <HealthCheckComponent
            accessToken="token"
            modelData={modelData}
            all_models_on_proxy={["id-alpha", "id-beta"]}
            getDisplayModelName={getDisplayModelName}
          />,
        );
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 0));
      });

      expect(mockLatestHealthChecksCall).toHaveBeenCalledWith("token");
      const healthyBadges = screen.getAllByText("healthy");
      const unhealthyBadges = screen.getAllByText("unhealthy");
      expect(healthyBadges.length).toBeGreaterThanOrEqual(1);
      expect(unhealthyBadges.length).toBeGreaterThanOrEqual(1);
    });

    it("should skip latest_health_checks entries whose key is not a known model id", async () => {
      const modelData = {
        data: [
          {
            model_name: "gpt-4",
            model_info: { id: "current-model-id" },
            litellm_model_name: "gpt-4",
          },
        ],
      };

      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "current-model-id": {
            status: "healthy",
            checked_at: "2024-01-15T10:00:00Z",
            error_message: null,
          },
          "deleted-or-unknown-id": {
            status: "unhealthy",
            checked_at: "2024-01-15T10:05:00Z",
            error_message: "Stale entry",
          },
        },
      });

      await act(async () => {
        render(
          <HealthCheckComponent
            accessToken="token"
            modelData={modelData}
            all_models_on_proxy={["current-model-id"]}
            getDisplayModelName={getDisplayModelName}
          />,
        );
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 0));
      });

      expect(screen.getByText("healthy")).toBeInTheDocument();
      expect(screen.queryByText("unhealthy")).not.toBeInTheDocument();
    });

    it("should not apply status when latest_health_checks key is model name not model id", async () => {
      const modelData = {
        data: [
          {
            model_name: "gpt-4",
            model_info: { id: "model-id-123" },
            litellm_model_name: "gpt-4",
          },
        ],
      };

      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "gpt-4": {
            status: "healthy",
            checked_at: "2024-01-15T10:00:00Z",
            error_message: null,
          },
        },
      });

      await act(async () => {
        render(
          <HealthCheckComponent
            accessToken="token"
            modelData={modelData}
            all_models_on_proxy={["model-id-123"]}
            getDisplayModelName={getDisplayModelName}
          />,
        );
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 0));
      });

      expect(screen.queryByText("healthy")).not.toBeInTheDocument();
      expect(screen.getByText("none")).toBeInTheDocument();
    });
  });
});
