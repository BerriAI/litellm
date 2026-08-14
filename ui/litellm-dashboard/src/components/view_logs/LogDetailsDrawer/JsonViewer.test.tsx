import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { JsonViewer } from "./JsonViewer";

describe("JsonViewer", () => {
  it("should render a placeholder and no tree when the log entry carries no payload", () => {
    render(<JsonViewer data={null} mode="formatted" />);

    expect(screen.getByText("No data")).toBeInTheDocument();
    expect(screen.queryByRole("tree")).not.toBeInTheDocument();
  });

  it("should render the payload as a tree exposing its keys", () => {
    render(<JsonViewer data={{ model: "claude-opus-4-5", stream: true }} mode="formatted" />);

    expect(screen.getByRole("tree")).toBeInTheDocument();
    expect(screen.getByText(/model/)).toBeInTheDocument();
    expect(screen.getByText(/stream/)).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });

  it("should treat an empty payload as data rather than showing the placeholder", () => {
    render(<JsonViewer data={{}} mode="formatted" />);

    expect(screen.getByRole("tree")).toBeInTheDocument();
    expect(screen.queryByText("No data")).not.toBeInTheDocument();
  });
});
