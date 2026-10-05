import { describe, expect, it } from "vitest";
import { initialSetupStep, readiness, type ReadinessInput } from "./readiness";

const base: ReadinessInput = {
  traces: { recorded: undefined, failed: false, disabled: false, checked: false },
  tracingConfigured: false,
  activity: undefined,
  activityError: null,
  connected: true,
  listError: null,
  investigations: 0,
};

describe("readiness", () => {
  it("counts traces confirmed from storage as recorded activity even when the activity check fails", () => {
    const failing = { ...base, activityError: new Error("Activity unavailable") };
    expect(readiness(failing).ready).toBe(false);
    const seen = { ...failing, traces: { ...base.traces, recorded: true } };
    expect(readiness(seen)).toMatchObject({ tracesReady: true, ready: true });
    expect(readiness({ ...seen, traces: { ...seen.traces, failed: true } }).tracesReady).toBe(false);
  });

  it("needs a connected worker and a healthy list on top of recorded activity", () => {
    const active = { ...base, activity: { traces: false, requests: true } };
    expect(readiness(active)).toMatchObject({ requestsReady: true, activityReady: true, ready: true });
    expect(readiness({ ...active, connected: false }).ready).toBe(false);
    expect(readiness({ ...active, listError: new Error("down") }).ready).toBe(false);
  });

  it("treats tracing as enabled once storage answers or the gateway reports it configured, unless storage says it is off", () => {
    expect(readiness(base).tracingEnabled).toBe(false);
    expect(readiness({ ...base, tracingConfigured: true }).tracingEnabled).toBe(true);
    expect(readiness({ ...base, traces: { ...base.traces, checked: true } }).tracingEnabled).toBe(true);
    const disabled = { ...base, tracingConfigured: true, traces: { ...base.traces, checked: true, disabled: true } };
    expect(readiness(disabled).tracingEnabled).toBe(false);
  });

  it("knows whether any investigation exists", () => {
    expect(readiness(base).hasInvestigations).toBe(false);
    expect(readiness({ ...base, investigations: 2 }).hasInvestigations).toBe(true);
  });
});

describe("initialSetupStep", () => {
  it("opens the first step that still needs work", () => {
    const offline = { ...base, connected: false };
    expect(initialSetupStep(readiness(offline))).toBe(0);
    expect(initialSetupStep(readiness({ ...offline, tracingConfigured: true }))).toBe(1);
    const recorded = { ...offline, activity: { traces: false, requests: true } };
    expect(initialSetupStep(readiness(recorded))).toBe(2);
    expect(initialSetupStep(readiness({ ...recorded, connected: true }))).toBe(3);
  });
});
