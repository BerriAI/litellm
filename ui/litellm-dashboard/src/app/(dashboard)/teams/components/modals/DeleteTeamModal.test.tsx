import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";
import { Team } from "@/components/key_team_helpers/key_list";
import DeleteTeamModal from "./DeleteTeamModal";

const makeTeam = (overrides: Partial<Team> = {}): Team => ({
  team_id: "team-1",
  team_alias: "Engineering",
  models: [],
  max_budget: null,
  budget_duration: null,
  tpm_limit: null,
  rpm_limit: null,
  organization_id: "org-1",
  created_at: "2024-01-01T00:00:00Z",
  keys: [],
  members_with_roles: [],
  spend: 0,
  ...overrides,
});

const renderModal = (props: Partial<Parameters<typeof DeleteTeamModal>[0]> = {}) => {
  const defaults = {
    teams: [makeTeam()],
    teamToDelete: "team-1",
    onCancel: vi.fn(),
    onConfirm: vi.fn(),
  };
  return render(<DeleteTeamModal {...defaults} {...props} />);
};

describe("DeleteTeamModal", () => {
  it("should render the title, team name label, and confirmation input", () => {
    renderModal();

    expect(screen.getByText("Delete Team")).toBeInTheDocument();
    expect(screen.getByText("Engineering")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Enter team name exactly")).toBeInTheDocument();
  });

  it("should render Cancel and Force Delete buttons", () => {
    renderModal();

    expect(screen.getByRole("button", { name: /^cancel$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /force delete/i })).toBeInTheDocument();
  });

  it("should not show the warning banner when the team has no keys", () => {
    renderModal({ teams: [makeTeam({ keys: [] })] });

    expect(screen.queryByText(/Warning/i)).not.toBeInTheDocument();
  });

  it("should show a warning with singular 'key' when the team has exactly 1 key", () => {
    const team = makeTeam({ keys: [{ token: "tok-1" } as any] });
    renderModal({ teams: [team] });

    expect(screen.getByText(/This team has 1 associated key\./)).toBeInTheDocument();
  });

  it("should show a warning with plural 'keys' when the team has multiple keys", () => {
    const team = makeTeam({
      keys: [{ token: "tok-1" } as any, { token: "tok-2" } as any, { token: "tok-3" } as any],
    });
    renderModal({ teams: [team] });

    expect(screen.getByText(/This team has 3 associated keys\./)).toBeInTheDocument();
  });

  it("should note that associated keys will also be deleted in the warning", () => {
    const team = makeTeam({ keys: [{ token: "tok-1" } as any] });
    renderModal({ teams: [team] });

    expect(screen.getByText(/Deleting the team will also delete all associated keys/)).toBeInTheDocument();
  });

  it("should disable Force Delete when the input is empty", () => {
    renderModal();

    expect(screen.getByRole("button", { name: /force delete/i })).toBeDisabled();
  });

  it("should keep Force Delete disabled when the input does not exactly match the team name", async () => {
    const user = userEvent.setup();
    renderModal();

    await user.type(screen.getByPlaceholderText("Enter team name exactly"), "engineer");

    expect(screen.getByRole("button", { name: /force delete/i })).toBeDisabled();
  });

  it("should enable Force Delete only after typing the exact team name (case-sensitive)", async () => {
    const user = userEvent.setup();
    renderModal();

    const input = screen.getByPlaceholderText("Enter team name exactly");

    await user.type(input, "Engineering");

    expect(screen.getByRole("button", { name: /force delete/i })).toBeEnabled();
  });

  it("should call onConfirm when Force Delete is clicked with a valid input", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderModal({ onConfirm });

    await user.type(screen.getByPlaceholderText("Enter team name exactly"), "Engineering");
    await user.click(screen.getByRole("button", { name: /force delete/i }));

    expect(onConfirm).toHaveBeenCalledTimes(1);
  });

  it("should not call onConfirm when Force Delete is clicked with an invalid input", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderModal({ onConfirm });

    // Button is disabled so click has no effect
    await user.click(screen.getByRole("button", { name: /force delete/i }));

    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("should call onCancel when the Cancel button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderModal({ onCancel });

    await user.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("should call onCancel when the Close button is clicked", async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    renderModal({ onCancel });

    await user.click(screen.getByRole("button", { name: /^close$/i }));

    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("should reset the confirmation input when Cancel is clicked", async () => {
    const user = userEvent.setup();
    renderModal();

    const input = screen.getByPlaceholderText("Enter team name exactly");
    await user.type(input, "Engineering");
    expect(input).toHaveValue("Engineering");

    await user.click(screen.getByRole("button", { name: /^cancel$/i }));

    expect(input).toHaveValue("");
  });

  it("should reset the confirmation input when the Close button is clicked", async () => {
    const user = userEvent.setup();
    renderModal();

    const input = screen.getByPlaceholderText("Enter team name exactly");
    await user.type(input, "Engineering");

    await user.click(screen.getByRole("button", { name: /^close$/i }));

    expect(input).toHaveValue("");
  });
});
