import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { credentialUpdateCall } from "../networking";
import EditLoggingCredentialModal from "./EditLoggingCredentialModal";

vi.mock("../networking", () => ({
  credentialUpdateCall: vi.fn(),
}));

vi.mock("../molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

// The access picker fetches teams/orgs through react-query; this test is about the
// PATCH body the modal builds, so the picker is stubbed out.
vi.mock("./AccessControlFields", () => ({ default: () => null }));

describe("EditLoggingCredentialModal", () => {
  const mockUpdate = vi.mocked(credentialUpdateCall);

  beforeEach(() => {
    vi.clearAllMocks();
    mockUpdate.mockResolvedValue({} as never);
  });

  // Regression: PATCH /credentials replaces credential_info wholesale. Sending only
  // { access } dropped credential_type/description, which stops the row being a logging
  // destination at all -- it vanishes from resolved_logging_exporters and exports nothing.
  it("resends the destination's whole credential_info with only access swapped", async () => {
    const user = userEvent.setup();
    render(
      <EditLoggingCredentialModal
        accessToken="tok"
        credentialName="dest-1"
        access={{ global: false, teams: ["team-a"], orgs: [] }}
        credentialInfo={{
          credential_type: "logging",
          description: "langfuse_otel",
          host: "https://collector.internal",
          access: { global: false, teams: ["team-a"], orgs: [] },
        }}
        open
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />,
    );

    await user.click(await screen.findByRole("button", { name: /save/i }));

    await waitFor(() => expect(mockUpdate).toHaveBeenCalled());
    const [, name, body] = mockUpdate.mock.calls[0];
    expect(name).toBe("dest-1");
    expect(body.credential_name).toBe("dest-1");
    // secrets are masked on read, so an access edit must not resend them
    expect(body.credential_values).toEqual({});
    // every sibling field survives the wholesale replace
    expect(body.credential_info).toMatchObject({
      credential_type: "logging",
      description: "langfuse_otel",
      host: "https://collector.internal",
    });
    expect(body.credential_info.access).toEqual({ global: false, teams: ["team-a"], orgs: [] });
  });
});
