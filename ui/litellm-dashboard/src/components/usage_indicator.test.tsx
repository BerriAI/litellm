import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import UsageIndicator from "./usage_indicator";

vi.mock("./networking", () => {
  return {
    getRemainingUsers: vi.fn(),
  };
});

import { getRemainingUsers } from "./networking";

describe("UsageIndicator", () => {
  it("does not show Near limit when users usage is below 80% (1/100 -> 1%)", async () => {
    (getRemainingUsers as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      total_users: 100,
      total_users_used: 1,
      total_users_remaining: 99,
      total_teams: null,
      total_teams_used: 0,
      total_teams_remaining: null,
    });

    const { queryByText, findByText } = render(<UsageIndicator accessToken={"token"} width={220} />);

    await findByText("Usage");

    expect(queryByText("Near limit")).toBeNull();
  });

  it("handles null totals shape by rendering nothing", async () => {
    (getRemainingUsers as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      total_users: null,
      total_teams: null,
      total_users_used: 520,
      total_teams_used: 4,
      total_teams_remaining: null,
      total_users_remaining: null,
    });

    const { container } = render(<UsageIndicator accessToken={"token"} width={220} />);

    await waitFor(() => {
      expect(container.firstChild).toBeNull();
    });
  });

  it("shows Near limit for Teams at 80% usage (4/5)", async () => {
    (getRemainingUsers as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      total_users: null,
      total_users_used: 0,
      total_users_remaining: null,
      total_teams: 5,
      total_teams_used: 4,
      total_teams_remaining: 1,
    });

    const { findByText, getByText } = render(<UsageIndicator accessToken={"token"} width={220} />);

    await findByText("Usage");

    // Teams section should show Near limit indicator
    expect(getByText("Teams")).toBeTruthy();
    expect(getByText("Near limit")).toBeTruthy();
  });

  it("shows Over limit for Users when usage exceeds 100% (105/100)", async () => {
    (getRemainingUsers as unknown as ReturnType<typeof vi.fn>).mockResolvedValue({
      total_users: 100,
      total_users_used: 105,
      total_users_remaining: -5,
      total_teams: null,
      total_teams_used: 0,
      total_teams_remaining: null,
    });

    const { findByText, getByText } = render(<UsageIndicator accessToken={"token"} width={220} />);

    await findByText("Usage");

    // Users section should show Over limit indicator
    expect(getByText("Users")).toBeTruthy();
    expect(getByText("Over limit")).toBeTruthy();
  });
});
