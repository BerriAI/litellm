import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AgentTracesTable } from "./AgentTracesTable";

const renderEmpty = (rangeEmpty: boolean) => {
  const onSetUpTracing = vi.fn();
  render(
    <AgentTracesTable
      traces={[]}
      isLoading={false}
      error={null}
      hasMore={false}
      onLoadMore={vi.fn()}
      onOpenTrace={vi.fn()}
      rangeEmpty={rangeEmpty}
      onSetUpTracing={onSetUpTracing}
    />,
  );
  return onSetUpTracing;
};

describe("AgentTracesTable empty state", () => {
  it.each([
    { rangeEmpty: true, message: "No runs in this time range" },
    { rangeEmpty: false, message: "No runs match these filters." },
  ])("offers tracing setup when rangeEmpty is $rangeEmpty", ({ rangeEmpty, message }) => {
    const onSetUpTracing = renderEmpty(rangeEmpty);
    expect(screen.getByText(message)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Set up tracing" }));
    expect(onSetUpTracing).toHaveBeenCalledOnce();
  });
});
