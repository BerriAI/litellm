import { renderHook } from "@testing-library/react";
import { act } from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useKeyboardNavigation } from "./useKeyboardNavigation";
import { LogEntry } from "../columns";

function makeLog(request_id: string): LogEntry {
  return {
    request_id,
    api_key: "key",
    team_id: "team",
    model: "gpt-4",
    model_id: "model-id",
    call_type: "completion",
    spend: 0,
    total_tokens: 0,
    prompt_tokens: 0,
    completion_tokens: 0,
    startTime: "2024-01-01T00:00:00Z",
    endTime: "2024-01-01T00:00:01Z",
    cache_hit: "False",
    messages: [],
    response: "",
  };
}

const logA = makeLog("log-a");
const logB = makeLog("log-b");
const logC = makeLog("log-c");
const allLogs = [logA, logB, logC];

describe("useKeyboardNavigation", () => {
  let onClose: ReturnType<typeof vi.fn>;
  let onSelectLog: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    onClose = vi.fn();
    onSelectLog = vi.fn();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const fireKey = (key: string) => {
    act(() => {
      window.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
    });
  };

  it("J key calls selectNextLog (moves to next/down in list)", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logA,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("j");
    expect(onSelectLog).toHaveBeenCalledWith(logB);
  });

  it("J (uppercase) key calls selectNextLog", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logA,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("J");
    expect(onSelectLog).toHaveBeenCalledWith(logB);
  });

  it("K key calls selectPreviousLog (moves to previous/up in list)", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logB,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("k");
    expect(onSelectLog).toHaveBeenCalledWith(logA);
  });

  it("K (uppercase) key calls selectPreviousLog", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logB,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("K");
    expect(onSelectLog).toHaveBeenCalledWith(logA);
  });

  it("J key does not go past the last log", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logC,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("j");
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("K key does not go before the first log", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logA,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("k");
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("Escape key calls onClose", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logA,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("Escape");
    expect(onClose).toHaveBeenCalled();
  });

  it("keys do nothing when drawer is closed", () => {
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: false,
        currentLog: logA,
        allLogs,
        onClose,
        onSelectLog,
      })
    );

    fireKey("j");
    fireKey("k");
    fireKey("Escape");
    expect(onSelectLog).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});
