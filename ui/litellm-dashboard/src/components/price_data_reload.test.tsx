import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { toast } from "@/lib/toast";
import {
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
  getModelCostMapSource,
  reloadModelCostMap,
  scheduleModelCostMapReload,
} from "./networking";
import PriceDataReload from "./price_data_reload";

vi.mock("./networking", () => ({
  cancelModelCostMapReload: vi.fn(),
  getModelCostMapReloadStatus: vi.fn(),
  getModelCostMapSource: vi.fn(),
  reloadModelCostMap: vi.fn(),
  scheduleModelCostMapReload: vi.fn(),
}));

const unscheduledStatus = { scheduled: false, interval_hours: null, last_run: null, next_run: null };
const scheduledStatus = {
  scheduled: true,
  interval_hours: 6,
  last_run: null,
  next_run: "2026-01-21T00:00:00Z",
};
const remoteSource = {
  source: "remote",
  url: "https://pricing.example.test/model_prices.json",
  is_env_forced: false,
  fallback_reason: null,
  model_count: 1234,
};

describe("PriceDataReload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getModelCostMapReloadStatus).mockResolvedValue(unscheduledStatus);
    vi.mocked(getModelCostMapSource).mockResolvedValue(remoteSource as never);
  });

  it("shows the pricing source and current reload status", async () => {
    render(<PriceDataReload accessToken="sk-test" />);

    expect(await screen.findByText("Pricing Data Source")).toBeInTheDocument();
    expect(screen.getByText("Remote")).toBeInTheDocument();
    expect(screen.getByText("1,234")).toBeInTheDocument();
    expect(screen.getByText("No periodic reload scheduled")).toBeInTheDocument();
  });

  it("confirms an immediate reload and refreshes dependent data", async () => {
    const user = userEvent.setup();
    const onReloadSuccess = vi.fn();
    vi.mocked(reloadModelCostMap).mockResolvedValue({ status: "success", models_count: 42 } as never);
    render(<PriceDataReload accessToken="sk-test" onReloadSuccess={onReloadSuccess} />);

    await user.click(screen.getByRole("button", { name: /Reload Price Data/ }));
    expect(screen.getByText("Hard Refresh Price Data")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Yes" }));

    await waitFor(() => expect(reloadModelCostMap).toHaveBeenCalledWith("sk-test"));
    expect(onReloadSuccess).toHaveBeenCalledTimes(1);
    expect(toast.success).toHaveBeenCalledWith("Price data reloaded successfully! 42 models updated.");
  });

  it("schedules periodic reloads using the selected interval", async () => {
    const user = userEvent.setup();
    vi.mocked(scheduleModelCostMapReload).mockResolvedValue({ status: "success" } as never);
    render(<PriceDataReload accessToken="sk-test" />);

    await user.click(screen.getByRole("button", { name: /Set Up Periodic Reload/ }));
    expect(screen.getByRole("dialog", { name: "Set Up Periodic Reload" })).toBeInTheDocument();
    const hours = screen.getByRole("spinbutton", { name: "Reload interval in hours" });
    await user.clear(hours);
    fireEvent.change(hours, { target: { value: "12" } });
    await user.click(screen.getByRole("button", { name: "Schedule" }));

    await waitFor(() => expect(scheduleModelCostMapReload).toHaveBeenCalledWith("sk-test", 12));
    expect(toast.success).toHaveBeenCalledWith("Periodic reload scheduled for every 12 hours");
  });

  it.each(["-1", "1.5", "169"])("should reject an invalid periodic reload interval of %s hours", async (interval) => {
    const user = userEvent.setup();
    render(<PriceDataReload accessToken="sk-test" />);

    await user.click(screen.getByRole("button", { name: /Set Up Periodic Reload/ }));
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: interval } });
    await user.click(screen.getByRole("button", { name: "Schedule" }));

    expect(scheduleModelCostMapReload).not.toHaveBeenCalled();
    expect(toast.fromError).toHaveBeenCalledWith("Hours must be a whole number between 1 and 168");
  });

  it.each([1, 168])("should schedule a periodic reload at the inclusive %s-hour boundary", async (interval) => {
    const user = userEvent.setup();
    vi.mocked(scheduleModelCostMapReload).mockResolvedValue({ status: "success" } as never);
    render(<PriceDataReload accessToken="sk-test" />);

    await user.click(screen.getByRole("button", { name: /Set Up Periodic Reload/ }));
    fireEvent.change(screen.getByRole("spinbutton"), { target: { value: String(interval) } });
    await user.click(screen.getByRole("button", { name: "Schedule" }));

    await waitFor(() => expect(scheduleModelCostMapReload).toHaveBeenCalledWith("sk-test", interval));
  });

  it("cancels an active periodic reload", async () => {
    const user = userEvent.setup();
    vi.mocked(getModelCostMapReloadStatus).mockResolvedValue(scheduledStatus);
    vi.mocked(cancelModelCostMapReload).mockResolvedValue({ status: "success" } as never);
    render(<PriceDataReload accessToken="sk-test" />);

    await user.click(await screen.findByRole("button", { name: /Cancel Periodic Reload/ }));

    await waitFor(() => expect(cancelModelCostMapReload).toHaveBeenCalledWith("sk-test"));
    expect(toast.success).toHaveBeenCalledWith("Periodic reload cancelled successfully");
  });
});
