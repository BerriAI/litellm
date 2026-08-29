import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import * as networking from "@/components/networking";
import PolicyTestPanel from "./policy_test_panel";

vi.mock("@/components/networking");

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ userId: "admin-user-id", userRole: "Admin", accessToken: "test-token" }),
}));

const RESOLVED = {
  effective_guardrails: ["pii-masking"],
  matched_policies: [{ policy_name: "policy-alpha", matched_via: "team_alias", guardrails_added: ["pii-masking"] }],
};

const setup = () => {
  const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  renderWithProviders(<PolicyTestPanel accessToken="test-token" />);
  return user;
};

const pickOption = async (user: ReturnType<typeof userEvent.setup>, label: string, option: string) => {
  await user.click(screen.getByLabelText(label));
  await user.click(await screen.findByTitle(option));
};

const simulate = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("button", { name: "Simulate" }));
};

describe("PolicyTestPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.teamListCall).mockResolvedValue([
      { team_alias: "team-a" },
      { team_alias: "team-b" },
    ] as unknown as Awaited<ReturnType<typeof networking.teamListCall>>);
    vi.mocked(networking.keyListCall).mockResolvedValue({ keys: [{ key_alias: "key-a" }] } as unknown as Awaited<
      ReturnType<typeof networking.keyListCall>
    >);
    vi.mocked(networking.modelAvailableCall).mockResolvedValue({ data: [{ id: "gpt-4" }] } as unknown as Awaited<
      ReturnType<typeof networking.modelAvailableCall>
    >);
    vi.mocked(networking.resolvePoliciesCall).mockResolvedValue(RESOLVED);
  });

  it("loads teams, keys and models for the dropdowns", async () => {
    setup();
    await waitFor(() => expect(networking.teamListCall).toHaveBeenCalledWith("test-token", null, "admin-user-id"));
    expect(networking.keyListCall).toHaveBeenCalledWith("test-token", null, null, null, null, null, 1, 100);
    expect(networking.modelAvailableCall).toHaveBeenCalledWith("test-token", "admin-user-id", "Admin");
  });

  it("sends an empty context when nothing has been picked", async () => {
    const user = setup();
    await simulate(user);
    await waitFor(() => expect(networking.resolvePoliciesCall).toHaveBeenCalledTimes(1));
    expect(networking.resolvePoliciesCall).toHaveBeenCalledWith("test-token", {});
  });

  it("sends only the fields that were filled in", async () => {
    const user = setup();
    await waitFor(() => expect(networking.teamListCall).toHaveBeenCalled());
    await pickOption(user, "Team Alias", "team-b");
    await pickOption(user, "Model", "gpt-4");

    await simulate(user);

    await waitFor(() => expect(networking.resolvePoliciesCall).toHaveBeenCalledTimes(1));
    expect(networking.resolvePoliciesCall).toHaveBeenCalledWith("test-token", {
      team_alias: "team-b",
      model: "gpt-4",
    });
  });

  it("turns each token-separated entry into its own tag", async () => {
    const user = setup();
    const tags = screen.getByLabelText("Tags");
    await user.click(tags);
    await user.type(tags, "prod-us,");
    await user.type(tags, "healthcare ");

    await simulate(user);

    await waitFor(() => expect(networking.resolvePoliciesCall).toHaveBeenCalledTimes(1));
    expect(networking.resolvePoliciesCall).toHaveBeenCalledWith("test-token", { tags: ["prod-us", "healthcare"] });
  });

  it("commits tag text still sitting in the box when the field loses focus", async () => {
    const user = setup();
    const tags = screen.getByLabelText("Tags");
    await user.click(tags);
    await user.type(tags, "prod-us");

    await simulate(user);

    await waitFor(() => expect(networking.resolvePoliciesCall).toHaveBeenCalledTimes(1));
    expect(networking.resolvePoliciesCall).toHaveBeenCalledWith("test-token", { tags: ["prod-us"] });
  });

  it("shows the placeholder before the first run, then the results", async () => {
    const user = setup();
    expect(screen.getByText("No simulation run yet")).toBeInTheDocument();

    await simulate(user);

    expect(await screen.findByText("Effective Guardrails")).toBeInTheDocument();
    expect(screen.getByText("Matched Policies")).toBeInTheDocument();
    expect(screen.getByText("policy-alpha")).toBeInTheDocument();
    expect(screen.getByText("team_alias")).toBeInTheDocument();
    expect(screen.queryByText("No simulation run yet")).not.toBeInTheDocument();
  });

  it("reports a failed resolve without results", async () => {
    vi.mocked(networking.resolvePoliciesCall).mockRejectedValue(new Error("boom"));
    const user = setup();

    await simulate(user);

    expect(await screen.findByText("Failed to resolve policies. Check the proxy logs.")).toBeInTheDocument();
  });

  it("clears both the picked context and the results on Reset", async () => {
    const user = setup();
    await waitFor(() => expect(networking.teamListCall).toHaveBeenCalled());
    await pickOption(user, "Team Alias", "team-a");
    await simulate(user);
    await screen.findByText("Effective Guardrails");

    await user.click(screen.getByRole("button", { name: "Reset" }));

    expect(screen.getByText("No simulation run yet")).toBeInTheDocument();
    await simulate(user);
    await waitFor(() => expect(networking.resolvePoliciesCall).toHaveBeenCalledTimes(2));
    expect(vi.mocked(networking.resolvePoliciesCall).mock.calls[1][1]).toEqual({});
  });

  it("does not resolve anything without an access token", async () => {
    const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
    renderWithProviders(<PolicyTestPanel accessToken={null} />);
    expect(networking.teamListCall).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Simulate" }));
    expect(networking.resolvePoliciesCall).not.toHaveBeenCalled();
  });
});
