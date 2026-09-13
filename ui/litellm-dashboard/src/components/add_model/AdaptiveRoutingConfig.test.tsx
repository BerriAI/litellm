import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import AdaptiveRoutingConfig from "./AdaptiveRoutingConfig";
import { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const adaptiveValue: ComplexityRouterConfigValue = {
  adaptive: true,
  adaptive_eligible: "all",
  tier_distance_penalty: 0.2,
};

describe("AdaptiveRoutingConfig", () => {
  it("keeps the tier distance penalty empty while it is being edited, then commits the new penalty", () => {
    const onChange = vi.fn();
    renderWithProviders(<AdaptiveRoutingConfig value={adaptiveValue} onChange={onChange} />);

    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "" } });

    expect(input).toHaveValue(null);
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.change(input, { target: { value: "0.5" } });

    expect(onChange).toHaveBeenCalledWith({ ...adaptiveValue, tier_distance_penalty: 0.5 });
  });

  it("restores the committed tier distance penalty after an empty field loses focus", () => {
    renderWithProviders(<AdaptiveRoutingConfig value={adaptiveValue} onChange={vi.fn()} />);

    const input = screen.getByRole("spinbutton");
    fireEvent.change(input, { target: { value: "" } });
    fireEvent.blur(input);

    expect(input).toHaveValue(0.2);
  });
});
