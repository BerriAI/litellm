import { fireEvent, renderWithProviders, screen } from "../../../tests/test-utils";
import { vi } from "vitest";
import type { ComplexityRouterConfigValue } from "./ComplexityRouterConfig";
import StallEscalationConfig, { stallEscalationBlockedReason } from "./StallEscalationConfig";

const tiers = { SIMPLE: "gpt-4o-mini", MEDIUM: "gpt-4o", COMPLEX: "claude-sonnet-4", REASONING: "o1-preview" };

const baseValue: ComplexityRouterConfigValue = {
  tiers,
  classifier_type: "heuristic",
};

const renderConfig = (value: Partial<ComplexityRouterConfigValue> = {}) => {
  const onChange = vi.fn();
  renderWithProviders(<StallEscalationConfig value={{ ...baseValue, ...value }} onChange={onChange} />);
  return onChange;
};

const toggle = () => screen.getByRole("switch", { name: "Escalate a stalled task to a stronger model" });

describe("stallEscalationBlockedReason", () => {
  it("blocks on session pinning, which replays a model instead of classifying", () => {
    expect(stallEscalationBlockedReason({ ...baseValue, session_affinity: true })).toContain("Classification Method");
  });

  it("blocks on user-turn classification, which skips the agent-loop turns a stall shows up in", () => {
    expect(stallEscalationBlockedReason({ ...baseValue, classification_mode: "user_turn" })).toContain("every request");
  });

  it("allows the default every-request router", () => {
    expect(stallEscalationBlockedReason(baseValue)).toBeNull();
  });
});

describe("StallEscalationConfig", () => {
  it("hides the knobs until the feature is turned on", () => {
    renderConfig();
    expect(toggle()).not.toBeChecked();
    expect(screen.queryByLabelText("Repeats before escalating")).not.toBeInTheDocument();
  });

  it("turning it on seeds both knobs so the saved config is explicit rather than half-set", () => {
    const onChange = renderConfig();
    fireEvent.click(toggle());
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        stall_escalation_enabled: true,
        stall_escalation_window: 6,
        stall_escalation_repeat_threshold: 3,
      }),
    );
  });

  it("turning it off clears all three keys, since the backend rejects them next to session pinning", () => {
    const onChange = renderConfig({
      stall_escalation_enabled: true,
      stall_escalation_window: 6,
      stall_escalation_repeat_threshold: 3,
    });
    fireEvent.click(toggle());
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({
        stall_escalation_enabled: undefined,
        stall_escalation_window: undefined,
        stall_escalation_repeat_threshold: undefined,
      }),
    );
  });

  it("raises the window to match a larger threshold, which could otherwise never be reached", () => {
    const onChange = renderConfig({
      stall_escalation_enabled: true,
      stall_escalation_window: 4,
      stall_escalation_repeat_threshold: 3,
    });
    fireEvent.change(screen.getByLabelText("Repeats before escalating"), { target: { value: "9" } });
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ stall_escalation_repeat_threshold: 9, stall_escalation_window: 9 }),
    );
  });

  it("holds the window at the threshold when someone types a smaller one", () => {
    const onChange = renderConfig({
      stall_escalation_enabled: true,
      stall_escalation_window: 6,
      stall_escalation_repeat_threshold: 3,
    });
    fireEvent.change(screen.getByLabelText("Recent calls examined"), { target: { value: "1" } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ stall_escalation_window: 3 }));
  });

  it("floors the threshold at 2, below which a single ordinary retry would escalate", () => {
    const onChange = renderConfig({ stall_escalation_enabled: true });
    fireEvent.change(screen.getByLabelText("Repeats before escalating"), { target: { value: "1" } });
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ stall_escalation_repeat_threshold: 2 }));
  });

  it("disables the toggle and says why when session pinning is on", () => {
    renderConfig({ session_affinity: true });
    expect(toggle()).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByText(/How often to classify/)).toBeInTheDocument();
  });

  it("hides the knobs when a blocker is switched on under an already-enabled router", () => {
    renderConfig({ stall_escalation_enabled: true, session_affinity: true });
    expect(screen.queryByLabelText("Repeats before escalating")).not.toBeInTheDocument();
  });

  it("still lets an already-on router turn it off once a blocker appears, which the save needs", () => {
    const onChange = renderConfig({ stall_escalation_enabled: true, session_affinity: true });
    expect(toggle()).not.toHaveAttribute("aria-disabled", "true");
    fireEvent.click(toggle());
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ stall_escalation_enabled: undefined }));
  });
});
