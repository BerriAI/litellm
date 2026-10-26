import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { dailyActivityExportCall } from "@/components/networking";
import type { AgentRow } from "../components/overview/agentCatalog";
import { MAX_CLIENT_MODEL_EXPORT_BYTES, useClientAgentModels } from "./useClientAgentModels";

vi.mock("@/components/networking", () => ({ dailyActivityExportCall: vi.fn() }));

const tag = "User-Agent: claude-cli/2.1.1 (external, claude-desktop-3p)";
const agent: AgentRow = {
  id: "claude-desktop",
  label: "Claude Desktop",
  description: "Desktop client",
  kind: "agent",
  logo: null,
  spend: 1,
  tokens: 100,
  requests: 2,
  users: 1,
  tags: [tag],
};
const records = [
  {
    date: "2026-10-22",
    entity_id: tag,
    model: "model-a",
    spend: 1,
    flat_cost: 0,
    prompt_tokens: 80,
    completion_tokens: 20,
    api_requests: 2,
    successful_requests: 2,
    failed_requests: 0,
    cache_read_input_tokens: 30,
    cache_creation_input_tokens: 10,
  },
];
const options = {
  accessToken: "test-token",
  agent,
  enabled: true,
  startTime: new Date("2026-10-22T07:00:00Z"),
  endTime: new Date("2026-10-23T06:59:59.999Z"),
};
const blob = (text: string) => Object.assign(new Blob([text]), { text: async () => text });
const jsonBlob = (value: unknown) => blob(JSON.stringify(value));

describe("useClientAgentModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should request the complete authenticated model export for exact local filtering", async () => {
    vi.mocked(dailyActivityExportCall).mockResolvedValue(jsonBlob(records));
    const { result } = renderHook(() => useClientAgentModels(options));
    await waitFor(() => expect(result.current.loading).toBe(false));
    const expectedRequest = {
      accessToken: options.accessToken,
      startTime: options.startTime,
      endTime: options.endTime,
      entityIds: null,
    };

    expect(dailyActivityExportCall).toHaveBeenCalledExactlyOnceWith(
      "tag",
      expectedRequest,
      "daily_with_models",
      "json",
    );
    expect(result.current.error).toBeNull();
    expect(result.current.results[0].metrics.spend).toBe(1);
  });

  it("should require the caller's enabled gate and authentication", () => {
    const { result, rerender } = renderHook(
      ({ enabled, accessToken }) => useClientAgentModels({ ...options, enabled, accessToken }),
      { initialProps: { enabled: false, accessToken: "test-token" as string | null } },
    );
    expect(result.current.results).toEqual([]);
    rerender({ enabled: true, accessToken: null });
    expect(result.current.loading).toBe(false);
    expect(dailyActivityExportCall).not.toHaveBeenCalled();
  });

  it("should reject an oversized export without parsing or displaying partial totals", async () => {
    const oversized = new Blob([new Uint8Array(MAX_CLIENT_MODEL_EXPORT_BYTES + 1)]);
    vi.mocked(dailyActivityExportCall).mockResolvedValue(oversized);
    const { result } = renderHook(() => useClientAgentModels(options));
    await waitFor(() => expect(result.current.error).toContain("shorter date range"));
    expect(result.current.results).toEqual([]);
  });

  it("should reject incomplete JSON rather than consume part of an export", async () => {
    vi.mocked(dailyActivityExportCall).mockResolvedValue(blob('[{"date":"2026-10-22"'));
    const { result } = renderHook(() => useClientAgentModels(options));
    await waitFor(() => expect(result.current.error).toBe("Could not load complete model usage. Try again."));
    expect(result.current.results).toEqual([]);
  });

  it("should clear loaded results synchronously when the range changes", async () => {
    vi.mocked(dailyActivityExportCall)
      .mockResolvedValueOnce(jsonBlob(records))
      .mockReturnValueOnce(new Promise(() => {}));
    const { result, rerender } = renderHook(({ startTime }) => useClientAgentModels({ ...options, startTime }), {
      initialProps: { startTime: options.startTime },
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    rerender({ startTime: new Date("2026-10-21T07:00:00Z") });
    expect(result.current.results).toEqual([]);
    expect(result.current.loading).toBe(true);
  });

  it("should ignore stale responses after the canonical tags change", async () => {
    let resolveFirst: ((value: Blob) => void) | undefined;
    const first = new Promise<Blob>((resolve) => {
      resolveFirst = resolve;
    });
    const changedAgent = { ...agent, tags: ["User-Agent: claude-desktop/3.0"] };
    const changedRows = records.map((row) => ({ ...row, entity_id: changedAgent.tags[0] }));
    vi.mocked(dailyActivityExportCall).mockReturnValueOnce(first).mockResolvedValueOnce(jsonBlob(changedRows));
    const { result, rerender } = renderHook(({ agent }) => useClientAgentModels({ ...options, agent }), {
      initialProps: { agent },
    });
    rerender({ agent: changedAgent });
    await waitFor(() => expect(result.current.loading).toBe(false));
    await act(async () => resolveFirst?.(jsonBlob(records)));
    expect(Object.keys(result.current.results[0].breakdown.entities)).toEqual(changedAgent.tags);
  });

  it("should clear loaded data immediately when the all-user panel is disabled", async () => {
    vi.mocked(dailyActivityExportCall).mockResolvedValue(jsonBlob(records));
    const { result, rerender } = renderHook(({ enabled }) => useClientAgentModels({ ...options, enabled }), {
      initialProps: { enabled: true },
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    rerender({ enabled: false });
    expect(result.current.results).toEqual([]);
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(false);
  });

  it("should retry a failed export with unchanged scope and clear its error immediately", async () => {
    vi.mocked(dailyActivityExportCall)
      .mockRejectedValueOnce(new Error("Export unavailable"))
      .mockResolvedValueOnce(jsonBlob(records));
    const { result } = renderHook(() => useClientAgentModels(options));
    await waitFor(() => expect(result.current.error).toBe("Export unavailable"));

    act(() => result.current.reload());
    expect(result.current.error).toBeNull();
    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    expect(dailyActivityExportCall).toHaveBeenCalledTimes(2);
    expect(vi.mocked(dailyActivityExportCall).mock.calls[1]).toEqual(vi.mocked(dailyActivityExportCall).mock.calls[0]);
  });

  it("should hide the previous model totals until a reload completes", async () => {
    vi.mocked(dailyActivityExportCall)
      .mockResolvedValueOnce(jsonBlob(records))
      .mockReturnValueOnce(new Promise(() => {}));
    const { result } = renderHook(() => useClientAgentModels(options));
    await waitFor(() => expect(result.current.results).toHaveLength(1));

    act(() => result.current.reload());
    expect(result.current.results).toEqual([]);
    expect(result.current.loading).toBe(true);
  });
});
