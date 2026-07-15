import React from "react";
import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import AgentVirtualKeys from "./agent_virtual_keys";
import type { KeyResponse } from "../key_team_helpers/key_list";

const makeKey = (overrides: Partial<KeyResponse>): KeyResponse =>
  ({
    token: "hash-abc123def456",
    key_alias: "agent-primary",
    key_name: "sk-...abcd",
    ...overrides,
  }) as unknown as KeyResponse;

describe("AgentVirtualKeys", () => {
  it("shows the empty state when the agent has no keys", () => {
    renderWithProviders(<AgentVirtualKeys keys={[]} isLoading={false} onKeyClick={vi.fn()} />);
    expect(screen.getByText("No virtual key assigned to this agent.")).toBeInTheDocument();
  });

  it("shows a loading state while keys are fetching", () => {
    renderWithProviders(<AgentVirtualKeys keys={[]} isLoading={true} onKeyClick={vi.fn()} />);
    expect(screen.getByText("Loading keys...")).toBeInTheDocument();
  });

  it("renders each attached key's alias and masked name", () => {
    const keys = [
      makeKey({ token: "hash-aaa", key_alias: "primary", key_name: "sk-...aaa" }),
      makeKey({ token: "hash-bbb", key_alias: "backup", key_name: "sk-...bbb" }),
    ];
    renderWithProviders(<AgentVirtualKeys keys={keys} isLoading={false} onKeyClick={vi.fn()} />);

    expect(screen.getByText("primary")).toBeInTheDocument();
    expect(screen.getByText("backup")).toBeInTheDocument();
    expect(screen.getByText("sk-...aaa")).toBeInTheDocument();
    expect(screen.getByText("sk-...bbb")).toBeInTheDocument();
  });

  it("fires onKeyClick with the full key when its Key ID is clicked", async () => {
    const onKeyClick = vi.fn();
    const key = makeKey({ token: "hash-clickme", key_alias: "clickable" });
    renderWithProviders(<AgentVirtualKeys keys={[key]} isLoading={false} onKeyClick={onKeyClick} />);

    await userEvent.click(screen.getByRole("button"));

    expect(onKeyClick).toHaveBeenCalledTimes(1);
    expect(onKeyClick).toHaveBeenCalledWith(key);
  });
});
