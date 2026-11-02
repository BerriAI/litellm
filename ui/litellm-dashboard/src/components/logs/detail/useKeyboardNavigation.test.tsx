import { fireEvent, renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { LogEntry } from "../types";
import { useKeyboardNavigation } from "./useKeyboardNavigation";

const logs = [{ request_id: "first" }, { request_id: "second" }] as LogEntry[];

describe("useKeyboardNavigation", () => {
  it("moves with plain J / K and ignores modified presses such as Cmd+J", () => {
    const onSelectLog = vi.fn();
    const props = { isOpen: true, currentLog: logs[0], allLogs: logs, onClose: vi.fn(), onSelectLog };
    renderHook(() => useKeyboardNavigation(props));

    fireEvent.keyDown(document.body, { key: "j", metaKey: true });
    fireEvent.keyDown(document.body, { key: "j", ctrlKey: true });
    fireEvent.keyDown(document.body, { key: "j", altKey: true });
    expect(onSelectLog).not.toHaveBeenCalled();

    fireEvent.keyDown(document.body, { key: "j" });
    expect(onSelectLog).toHaveBeenCalledExactlyOnceWith(logs[1]);
  });
});
