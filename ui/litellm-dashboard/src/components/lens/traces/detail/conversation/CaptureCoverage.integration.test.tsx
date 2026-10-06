import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { CaptureCoverage } from "./CaptureCoverage";
import type { Trace } from "../../types";

const coverage: NonNullable<Trace["capture"]> = {
  actors: [
    {
      actor_id: "root",
      name: "Root",
      llm_calls: 2,
      tool_calls: 1,
      reply_events: 1,
      model_outputs: 0,
      content_events: 1,
    },
    {
      actor_id: "child",
      name: "Reader",
      llm_calls: 3,
      tool_calls: 2,
      reply_events: 0,
      model_outputs: 0,
      content_events: 0,
    },
  ],
  content_events: 2,
  unassigned_events: 1,
  warning_events: 1,
};

describe("CaptureCoverage", () => {
  it("shows missing child capture even when the root reply was recorded", () => {
    render(<CaptureCoverage capture={coverage} />);
    expect(screen.getByText("No reply recorded yet for Reader.")).toBeVisible();
    expect(screen.getByText(/More records may still arrive/)).toBeVisible();
    expect(screen.getByText("1 content events have unconfirmed actor ownership.")).toBeVisible();
    expect(screen.getByText(/1 records carry capture warnings/)).toBeVisible();
  });

  it("accepts recorded model output as reply evidence", () => {
    const capture = { ...coverage, actors: coverage.actors.map((actor) => ({ ...actor, model_outputs: 1 })) };
    render(<CaptureCoverage capture={capture} />);
    expect(screen.queryByText(/No reply recorded yet/)).not.toBeInTheDocument();
  });
});
