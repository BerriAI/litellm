import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { MemoryRow } from "@/components/networking";

import { MemoryDetailDrawer } from "./MemoryDetailDrawer";

const makeMemory = (overrides: Partial<MemoryRow> = {}): MemoryRow => ({
  memory_id: "mem-1",
  key: "user:profile",
  value: "The user prefers concise answers.",
  metadata: null,
  user_id: "user-42",
  team_id: "team-7",
  created_at: "2024-05-01T12:00:00Z",
  updated_at: "2024-05-02T12:00:00Z",
  created_by: "alice",
  updated_by: "bob",
  ...overrides,
});

describe("MemoryDetailDrawer", () => {
  it("renders nothing until a row is selected", () => {
    render(<MemoryDetailDrawer row={null} onClose={vi.fn()} />);

    expect(screen.queryByText("Memory ID")).not.toBeInTheDocument();
    expect(screen.queryByText("Value")).not.toBeInTheDocument();
  });

  it("shows the selected row's key, identifiers and value", () => {
    render(<MemoryDetailDrawer row={makeMemory()} onClose={vi.fn()} />);

    expect(screen.getByText("user:profile")).toBeInTheDocument();
    expect(screen.getByText("Memory ID")).toBeInTheDocument();
    expect(screen.getByText("mem-1")).toBeInTheDocument();
    expect(screen.getByText("User ID")).toBeInTheDocument();
    expect(screen.getByText("user-42")).toBeInTheDocument();
    expect(screen.getByText("Team ID")).toBeInTheDocument();
    expect(screen.getByText("team-7")).toBeInTheDocument();
    expect(screen.getByText("Value")).toBeInTheDocument();
    expect(screen.getByText("The user prefers concise answers.")).toBeInTheDocument();
  });

  it("falls back to a dash for a memory with no owning user or team", () => {
    render(<MemoryDetailDrawer row={makeMemory({ user_id: null, team_id: null })} onClose={vi.fn()} />);

    expect(screen.getAllByText("-")).toHaveLength(2);
    expect(screen.queryByText("user-42")).not.toBeInTheDocument();
  });

  it("omits the metadata block when the row carries no metadata", () => {
    render(<MemoryDetailDrawer row={makeMemory({ metadata: null })} onClose={vi.fn()} />);

    expect(screen.queryByText("Metadata")).not.toBeInTheDocument();
  });

  it("pretty-prints metadata as JSON when present", () => {
    render(<MemoryDetailDrawer row={makeMemory({ metadata: { tags: ["example"] } })} onClose={vi.fn()} />);

    expect(screen.getByText("Metadata")).toBeInTheDocument();
    expect(screen.getByText('{ "tags": [ "example" ] }')).toBeInTheDocument();
  });

  it("attributes the created and updated timestamps to their actors", () => {
    render(<MemoryDetailDrawer row={makeMemory()} onClose={vi.fn()} />);

    expect(screen.getByText(/^Created .* by alice$/)).toBeInTheDocument();
    expect(screen.getByText(/^Updated .* by bob$/)).toBeInTheDocument();
  });

  it("renders an em dash for a timestamp the backend did not send", () => {
    render(<MemoryDetailDrawer row={makeMemory({ created_at: undefined, created_by: undefined })} onClose={vi.fn()} />);

    expect(screen.getByText("Created —")).toBeInTheDocument();
  });

  it("closes through the close control", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(<MemoryDetailDrawer row={makeMemory()} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /close/i }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
