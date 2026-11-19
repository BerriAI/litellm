import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import AdditionalAuthProfilesSection from "./AdditionalAuthProfilesSection";
import type { AdditionalAuthProfile } from "@/components/mcp_tools/types";
import { draftFromProfile, newProfileDraft, type AdditionalAuthDraft } from "./additionalAuthProfiles";

const STORED_PROFILE: AdditionalAuthProfile = {
  id: "tickets-bots",
  auth_type: "oauth2",
  oauth2_flow: "client_credentials",
  client_id: null,
  client_secret: null,
  token_url: "https://auth.example.com/token",
};
const loaded = draftFromProfile(STORED_PROFILE);

const EditableSection: React.FC<{ readonly initial: readonly AdditionalAuthDraft[] }> = ({ initial }) => {
  const [drafts, setDrafts] = React.useState(initial);
  return <AdditionalAuthProfilesSection value={drafts} onChange={setDrafts} />;
};

describe("AdditionalAuthProfilesSection", () => {
  it("adds a blank client-credentials row", () => {
    const onChange = vi.fn();
    render(<AdditionalAuthProfilesSection value={[]} onChange={onChange} />);

    fireEvent.click(screen.getByRole("button", { name: "Add auth profile" }));

    expect(onChange).toHaveBeenCalledWith([{ ...newProfileDraft(), rowKey: expect.any(String) }]);
  });

  it("keeps a saved profile's id fixed and never fills the secret", () => {
    render(<AdditionalAuthProfilesSection value={[loaded]} onChange={vi.fn()} />);

    expect(screen.getByLabelText("Profile ID")).toBeDisabled();
    expect(screen.getByLabelText("Profile ID")).toHaveValue("tickets-bots");
    expect(screen.getByLabelText("Client secret")).toHaveValue("");
    expect(screen.getByLabelText("Client secret")).toHaveAttribute("placeholder", "Leave blank to keep existing");
  });

  it("shows a keep hint on a loaded profile's client id", () => {
    render(<AdditionalAuthProfilesSection value={[loaded]} onChange={vi.fn()} />);

    expect(screen.getByLabelText("Client ID")).toHaveValue("");
    expect(screen.getByLabelText("Client ID")).toHaveAttribute("placeholder", "Leave blank to keep existing");
  });

  it("shows no keep hint on a new profile's client id, which has nothing stored", () => {
    render(<AdditionalAuthProfilesSection value={[newProfileDraft()]} onChange={vi.fn()} />);

    expect(screen.getByLabelText("Client ID")).not.toHaveAttribute("placeholder", "Leave blank to keep existing");
  });

  it("offers only the two token endpoint auth methods", async () => {
    const onChange = vi.fn();
    render(<AdditionalAuthProfilesSection value={[loaded]} onChange={onChange} />);

    await userEvent.click(screen.getByLabelText("Token endpoint auth method"));
    const options = await screen.findAllByRole("option");
    expect(options.map((option) => option.textContent)).toEqual([
      "Not set",
      "client_secret_basic",
      "client_secret_post",
    ]);

    await userEvent.click(screen.getByRole("option", { name: "client_secret_post" }));
    expect(onChange).toHaveBeenCalledWith([{ ...loaded, token_endpoint_auth_method: "client_secret_post" }]);
  });

  it("removes only the row whose button was clicked", () => {
    const onChange = vi.fn();
    const other = draftFromProfile({ ...STORED_PROFILE, id: "wiki-bots" });
    render(<AdditionalAuthProfilesSection value={[loaded, other]} onChange={onChange} />);

    fireEvent.click(screen.getAllByRole("button", { name: "Remove auth profile" })[0]);

    expect(onChange).toHaveBeenCalledWith([other]);
  });

  it("keeps a revealed secret with its own row when an earlier row is removed", async () => {
    render(<EditableSection initial={[newProfileDraft(), newProfileDraft()]} />);

    await userEvent.click(screen.getAllByRole("button", { name: "Show password" })[1]);
    await userEvent.click(screen.getAllByRole("button", { name: "Remove auth profile" })[0]);

    expect(screen.getByLabelText("Client secret")).toHaveAttribute("type", "text");
  });

  it("flags a new profile's empty id on its field, as submit would refuse it", () => {
    render(<AdditionalAuthProfilesSection value={[newProfileDraft()]} onChange={vi.fn()} />);

    expect(screen.getByLabelText("Profile ID")).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByText("Use 1 to 64 letters, digits, hyphens or underscores.")).toBeInTheDocument();
  });

  it("tells the admin which id is reserved or already used", () => {
    const duplicate = { ...newProfileDraft(), id: "TICKETS-Bots" };
    render(<AdditionalAuthProfilesSection value={[loaded, duplicate]} onChange={vi.fn()} />);

    expect(screen.getAllByText("Profile IDs must be unique, ignoring case.")).toHaveLength(2);
  });
});
