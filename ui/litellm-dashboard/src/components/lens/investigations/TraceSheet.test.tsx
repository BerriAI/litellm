import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { RunSelection } from "@/components/view_logs/TraceView/traceRouting";

import { TraceSheet } from "./TraceSheet";

vi.mock("@/components/view_logs/TraceView/TraceDrawer", () => ({
  RunView: ({ selection }: { selection: RunSelection }) => <div data-testid="run-view">{selection.spanId}</div>,
}));

describe("TraceSheet", () => {
  it("opens another finding's evidence on its own step", () => {
    const props = { open: true, traceId: "trace-1", accessToken: "t", onClose: vi.fn() };
    const { rerender } = render(<TraceSheet {...props} initialSpanId="step-a" />);
    expect(screen.getByTestId("run-view")).toHaveTextContent("step-a");
    rerender(<TraceSheet {...props} initialSpanId="step-b" />);
    expect(screen.getByTestId("run-view")).toHaveTextContent("step-b");
  });
});
