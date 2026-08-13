/* @vitest-environment jsdom */
import type { PaginationState } from "@tanstack/react-table";
import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import HealthCheckComponent from "./HealthCheckComponent";

const mockIndividualModelHealthCheckCall = vi.fn();
const mockLatestHealthChecksCall = vi.fn();

vi.mock("../networking", () => ({
  individualModelHealthCheckCall: (...args: unknown[]) => mockIndividualModelHealthCheckCall(...args),
  latestHealthChecksCall: (...args: unknown[]) => mockLatestHealthChecksCall(...args),
}));

const getDisplayModelName = (model: { model_name?: string }) => model.model_name ?? "";

const makeModel = (id: string, name = "gpt-4") => ({
  model_name: name,
  model_info: { id },
  litellm_model_name: name,
});

interface HarnessProps {
  modelData: { data: ReturnType<typeof makeModel>[] };
  allModelsOnProxy: string[];
  rowCount?: number;
  onPageIndexChange?: (pageIndex: number) => void;
}

/** Holds pagination state so page changes exercise the real controlled wiring. */
function Harness({ modelData, allModelsOnProxy, rowCount = 1, onPageIndexChange }: HarnessProps) {
  const [pagination, setPagination] = useState<PaginationState>({ pageIndex: 0, pageSize: 50 });

  return (
    <>
      <span data-testid="page-index">{pagination.pageIndex}</span>
      <HealthCheckComponent
        accessToken="token"
        modelData={modelData}
        all_models_on_proxy={allModelsOnProxy}
        getDisplayModelName={getDisplayModelName}
        pagination={pagination}
        onPaginationChange={(updater) => {
          setPagination((previous) => {
            const next = typeof updater === "function" ? updater(previous) : updater;
            onPageIndexChange?.(next.pageIndex);
            return next;
          });
        }}
        rowCount={rowCount}
      />
    </>
  );
}

const renderHealthCheck = async (props: HarnessProps) => {
  await act(async () => {
    render(<Harness {...props} />);
  });
  await act(async () => {
    await new Promise((r) => setTimeout(r, 0));
  });
};

describe("HealthCheckComponent", () => {
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
    await renderHealthCheck({ modelData: { data: [makeModel("deployment-1")] }, allModelsOnProxy: ["deployment-1"] });

    expect(screen.getByText("Model Health Status")).toBeInTheDocument();
    expect(
      screen.getByText("Run health checks on individual models to verify they are working correctly"),
    ).toBeInTheDocument();
  });

  it("should call individualModelHealthCheckCall with model id when run health check is triggered", async () => {
    render(
      <Harness modelData={{ data: [makeModel("deployment-abc-123")] }} allModelsOnProxy={["deployment-abc-123"]} />,
    );

    const runButtons = screen.getAllByTestId("run-health-check-btn");
    expect(runButtons.length).toBeGreaterThanOrEqual(1);

    await act(async () => {
      runButtons[0].click();
    });
    await act(async () => {
      await new Promise((r) => setTimeout(r, 50));
    });

    expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "deployment-abc-123");
    expect(mockIndividualModelHealthCheckCall).not.toHaveBeenCalledWith("token", "gpt-4");
  });

  it("should page through results with the shared pagination footer", async () => {
    const onPageIndexChange = vi.fn();
    await renderHealthCheck({
      modelData: { data: [makeModel("deployment-1")] },
      allModelsOnProxy: ["deployment-1"],
      rowCount: 75,
      onPageIndexChange,
    });

    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-50 of 75");

    const user = userEvent.setup();
    await user.click(screen.getByTestId("pagination-next"));

    expect(onPageIndexChange).toHaveBeenCalledWith(1);
    expect(screen.getByTestId("page-index")).toHaveTextContent("1");
  });

  describe("row selection drives the bulk run", () => {
    const twoModels = { data: [makeModel("id-alpha", "alpha"), makeModel("id-beta", "beta")] };
    const bothIds = ["id-alpha", "id-beta"];

    it("runs only the selected models and labels the button accordingly", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");

      await user.click(screen.getByTestId("datatable-select-row-id-beta"));
      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run Selected Checks");

      await act(async () => {
        screen.getByTestId("run-health-checks").click();
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-beta");
      expect(mockIndividualModelHealthCheckCall).not.toHaveBeenCalledWith("token", "id-alpha");
    });

    it("falls back to every model on the page when nothing is selected", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });

      await act(async () => {
        screen.getByTestId("run-health-checks").click();
      });
      await act(async () => {
        await new Promise((r) => setTimeout(r, 50));
      });

      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-alpha");
      expect(mockIndividualModelHealthCheckCall).toHaveBeenCalledWith("token", "id-beta");
    });

    it("treats a full page selection as running everything", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      await user.click(screen.getByTestId("datatable-select-all"));

      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");
    });

    it("clears the selection from the Clear Selection button", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 2 });
      const user = userEvent.setup();

      expect(screen.queryByTestId("clear-health-selection")).not.toBeInTheDocument();

      await user.click(screen.getByTestId("datatable-select-row-id-alpha"));
      await user.click(screen.getByTestId("clear-health-selection"));

      expect(screen.getByTestId("datatable-select-row-id-alpha")).toHaveAttribute("aria-checked", "false");
      expect(screen.getByTestId("run-health-checks")).toHaveTextContent("Run All Checks");
    });

    // The pager swaps the underlying rows, so a carried-over selection would target
    // models that are no longer on screen.
    it("wipes the selection when the page changes", async () => {
      await renderHealthCheck({ modelData: twoModels, allModelsOnProxy: bothIds, rowCount: 120 });
      const user = userEvent.setup();

      await user.click(screen.getByTestId("datatable-select-row-id-alpha"));
      expect(screen.getByTestId("clear-health-selection")).toBeInTheDocument();

      await user.click(screen.getByTestId("pagination-next"));

      expect(screen.queryByTestId("clear-health-selection")).not.toBeInTheDocument();
      expect(screen.getByTestId("datatable-select-row-id-alpha")).toHaveAttribute("aria-checked", "false");
    });
  });

  describe("latest_health_checks keyed by model id", () => {
    it("should show status from latest_health_checks when keys match model ids", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "id-alpha": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
          "id-beta": { status: "unhealthy", checked_at: "2024-01-15T10:05:00Z", error_message: "Connection failed" },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("id-alpha"), makeModel("id-beta")] },
        allModelsOnProxy: ["id-alpha", "id-beta"],
        rowCount: 2,
      });

      expect(mockLatestHealthChecksCall).toHaveBeenCalledWith("token");
      expect(screen.getAllByText("healthy").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("unhealthy").length).toBeGreaterThanOrEqual(1);
    });

    it("should skip latest_health_checks entries whose key is not a known model id", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "current-model-id": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
          "deleted-or-unknown-id": { status: "unhealthy", checked_at: "2024-01-15T10:05:00Z", error_message: "Stale" },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("current-model-id")] },
        allModelsOnProxy: ["current-model-id"],
      });

      expect(screen.getByText("healthy")).toBeInTheDocument();
      expect(screen.queryByText("unhealthy")).not.toBeInTheDocument();
    });

    it("should not apply status when latest_health_checks key is model name not model id", async () => {
      mockLatestHealthChecksCall.mockResolvedValue({
        latest_health_checks: {
          "gpt-4": { status: "healthy", checked_at: "2024-01-15T10:00:00Z", error_message: null },
        },
      });

      await renderHealthCheck({
        modelData: { data: [makeModel("model-id-123")] },
        allModelsOnProxy: ["model-id-123"],
      });

      expect(screen.queryByText("healthy")).not.toBeInTheDocument();
      expect(screen.getByText("none")).toBeInTheDocument();
    });
  });
});
