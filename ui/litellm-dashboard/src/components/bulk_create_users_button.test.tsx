import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, it, expect, vi } from "vitest";
import Papa from "papaparse";
import type { ParseConfig, ParseResult } from "papaparse";
import BulkCreateUsersButton from "./bulk_create_users_button";
import { getProxyUISettings, invitationCreateCall, userCreateCall } from "./networking";

vi.mock("papaparse", () => ({
  default: {
    parse: vi.fn(),
    unparse: vi.fn(),
  },
}));

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({
    PROXY_BASE_URL: null,
    PROXY_LOGOUT_URL: null,
    DEFAULT_TEAM_DISABLED: false,
    SSO_ENABLED: false,
  }),
}));

vi.mock("./molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

describe("BulkCreateUsersButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.history.pushState({}, "", "/");

    vi.mocked(getProxyUISettings).mockResolvedValue({
      PROXY_BASE_URL: null,
      PROXY_LOGOUT_URL: null,
      DEFAULT_TEAM_DISABLED: false,
      SSO_ENABLED: false,
    });
    vi.mocked(userCreateCall).mockResolvedValue({
      user_id: "user-1",
      key: "key-1",
    });
    vi.mocked(invitationCreateCall).mockResolvedValue({
      id: "invite-1",
    });
    vi.mocked(Papa.parse).mockImplementation((_file: unknown, config?: ParseConfig<string[]>) => {
      const parsedCsv = {
        data: [
          ["user_email", "user_role"],
          ["person@example.com", "internal_user"],
        ],
        errors: [],
        meta: {},
      } as ParseResult<string[]>;

      config?.complete?.(parsedCsv);
      return undefined as ReturnType<typeof Papa.parse>;
    });
  });

  it("should render", () => {
    const { getByText } = render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });

  it("keeps the dashboard subpath in generated invitation links", async () => {
    window.history.pushState({}, "", "/litellm/ui");
    const user = userEvent.setup();
    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);

    await user.click(screen.getByText("+ Bulk Invite Users"));

    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input).not.toBeNull();

    const csv = new File(["user_email,user_role\nperson@example.com,internal_user\n"], "users.csv", {
      type: "text/csv",
    });
    await user.upload(input, csv);

    await screen.findByText("person@example.com");
    await user.click(screen.getAllByRole("button", { name: /Create 1 Users/i })[0]);

    await waitFor(() => expect(invitationCreateCall).toHaveBeenCalledWith("test-token", "user-1"));
    expect(
      await screen.findByText(`${window.location.origin}/litellm/ui?invitation_id=invite-1`),
    ).toBeInTheDocument();
  });
});
