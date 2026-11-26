import { describe, expect, it, vi } from "vitest";
import { fireEvent, renderWithProviders, screen, within } from "../../../tests/test-utils";
import AlertThresholds from "./AlertThresholds";

const thresholds = {
  spend: { multiplier: 3, minimum_increase: 50 },
  traffic: { multiplier: 10, minimum_increase: 1000 },
  latency: { multiplier: 3, minimum_increase: 500 },
  reliability: { multiplier: 3, minimum_increase: 0.02 },
};

describe("AlertThresholds", () => {
  it("saves exactly the values on screen after the thresholds are refetched mid-edit", async () => {
    const save = vi.fn().mockResolvedValue(undefined);
    const props = { enterprise: true, readOnly: false, pending: false, save };
    const { rerender } = renderWithProviders(<AlertThresholds thresholds={thresholds} {...props} />);
    fireEvent.click(screen.getByRole("button", { name: "Edit thresholds" }));
    const multiplier = (group: string) =>
      within(screen.getByRole("group", { name: group })).getByRole("spinbutton", { name: "Times the usual value" });
    fireEvent.change(multiplier("Spend"), { target: { value: "4" } });
    rerender(
      <AlertThresholds thresholds={{ ...thresholds, traffic: { multiplier: 7, minimum_increase: 900 } }} {...props} />,
    );
    const shownTraffic = (multiplier("Traffic") as HTMLInputElement).valueAsNumber;
    fireEvent.click(screen.getByRole("button", { name: "Save thresholds" }));
    expect(save).toHaveBeenCalledTimes(1);
    expect(save.mock.calls[0][0].spend.multiplier).toBe(4);
    expect(save.mock.calls[0][0].traffic.multiplier).toBe(shownTraffic);
  });
});
