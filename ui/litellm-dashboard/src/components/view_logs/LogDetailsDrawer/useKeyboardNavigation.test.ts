import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { useKeyboardNavigation } from "./useKeyboardNavigation";
import { LogEntry } from "../columns";

// Three rows is enough to exercise prev/next at head, middle, and tail.
const makeLogs = (): LogEntry[] =>
  [
    { request_id: "req-a", call_type: "completion", model: "gpt-4o", startTime: "", endTime: "" },
    { request_id: "req-b", call_type: "completion", model: "gpt-4o", startTime: "", endTime: "" },
    { request_id: "req-c", call_type: "completion", model: "gpt-4o", startTime: "", endTime: "" },
  ] as unknown as LogEntry[];

const dispatchKey = (key: string, opts: { shift?: boolean } = {}) => {
  window.dispatchEvent(new KeyboardEvent("keydown", { key, shiftKey: opts.shift ?? false }));
};

describe("useKeyboardNavigation", () => {
  it("J selects the next log within the current page", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[0],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
      }),
    );

    dispatchKey("j");
    expect(onSelectLog).toHaveBeenCalledWith(logs[1]);
  });

  it("K selects the previous log within the current page", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[2],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
      }),
    );

    dispatchKey("k");
    expect(onSelectLog).toHaveBeenCalledWith(logs[1]);
  });

  it("J at the end of the page is a no-op (does not page forward)", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    const onNextPage = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[2],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
        onNextPage,
      }),
    );

    dispatchKey("j");
    expect(onSelectLog).not.toHaveBeenCalled();
    expect(onNextPage).not.toHaveBeenCalled();
  });

  it("Shift+J calls onNextPage and does NOT walk to the next row", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    const onNextPage = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[0],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
        onNextPage,
      }),
    );

    // With shift held, the browser delivers e.key === "J" (uppercase). The
    // hook must route to onNextPage and skip selectNextLog so the two paths
    // stay mutually exclusive.
    dispatchKey("J", { shift: true });
    expect(onNextPage).toHaveBeenCalledTimes(1);
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("Shift+K calls onPreviousPage and does NOT walk to the previous row", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    const onPreviousPage = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[2],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
        onPreviousPage,
      }),
    );

    dispatchKey("K", { shift: true });
    expect(onPreviousPage).toHaveBeenCalledTimes(1);
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("Shift+J without an onNextPage handler is a safe no-op", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: logs[0],
        allLogs: logs,
        onClose: vi.fn(),
        onSelectLog,
      }),
    );

    expect(() => dispatchKey("J", { shift: true })).not.toThrow();
    // Critically, plain row-walk must not fire either — Shift+J is reserved
    // for paging even when no handler is wired.
    expect(onSelectLog).not.toHaveBeenCalled();
  });

  it("Escape calls onClose", () => {
    const onClose = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: true,
        currentLog: makeLogs()[0],
        allLogs: makeLogs(),
        onClose,
        onSelectLog: vi.fn(),
      }),
    );

    dispatchKey("Escape");
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("ignores keys when isOpen is false", () => {
    const logs = makeLogs();
    const onSelectLog = vi.fn();
    const onClose = vi.fn();
    renderHook(() =>
      useKeyboardNavigation({
        isOpen: false,
        currentLog: logs[0],
        allLogs: logs,
        onClose,
        onSelectLog,
      }),
    );

    dispatchKey("j");
    dispatchKey("Escape");
    expect(onSelectLog).not.toHaveBeenCalled();
    expect(onClose).not.toHaveBeenCalled();
  });
});
