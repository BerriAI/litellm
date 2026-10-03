import { fireEvent, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { LogEntry } from "../columns";
import { useKeyboardNavigation } from "./useKeyboardNavigation";

const logs = [{ request_id: "first" }, { request_id: "second" }] as LogEntry[];

describe("useKeyboardNavigation", () => {
  it("moves with plain J / K and ignores modified presses such as Cmd+J", () => {
    const onSelectLog = vi.fn();
    const props = { isOpen: true, currentLog: logs[0], allLogs: logs, onClose: vi.fn(), onSelectLog };
    renderHook(() => useKeyboardNavigation(props));

    fireEvent.keyDown(window, { key: "j", metaKey: true });
    fireEvent.keyDown(window, { key: "j", ctrlKey: true });
    fireEvent.keyDown(window, { key: "j", altKey: true });
    expect(onSelectLog).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "j" });
    expect(onSelectLog).toHaveBeenCalledExactlyOnceWith(logs[1]);
  });
});
