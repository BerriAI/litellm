import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../../../../tests/test-utils";
import { BulkInviteButton } from "./BulkInviteButton";

const { userCreateCall, invitationCreateCall } = vi.hoisted(() => ({
  userCreateCall: vi.fn(),
  invitationCreateCall: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  userCreateCall,
  invitationCreateCall,
  getProxyBaseUrl: () => "",
  getUiPageUrl: (url: string) => url,
  getProxyUISettings: vi.fn().mockResolvedValue({ SSO_ENABLED: false }),
}));

const csvFile = (text: string) => new File([text], "users.csv", { type: "text/csv" });

const readBlob = (blob: Blob) =>
  new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.readAsText(blob);
  });

const openDialog = async (onUsersCreated?: () => void) => {
  const user = userEvent.setup();
  renderWithProviders(
    <BulkInviteButton
      accessToken="test-token"
      teams={[{ team_id: "team-a" }]}
      possibleUIRoles={null}
      onUsersCreated={onUsersCreated}
    />,
  );
  await user.click(screen.getByRole("button", { name: "+ Bulk Invite Users" }));
  return user;
};

const fileInput = () => screen.getByLabelText(/drag and drop your csv file here/i);

describe("Bulk invite dialog", () => {
  const originalCreateObjectURL = URL.createObjectURL;
  const originalRevokeObjectURL = URL.revokeObjectURL;
  const createObjectURL = vi.fn((_blob: Blob) => "blob:download");
  const revokeObjectURL = vi.fn();
  let anchorClick: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    vi.clearAllMocks();
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  });

  afterEach(() => {
    anchorClick.mockRestore();
    URL.createObjectURL = originalCreateObjectURL;
    URL.revokeObjectURL = originalRevokeObjectURL;
  });

  const lastDownload = async () => {
    const anchor = anchorClick.mock.contexts.at(-1) as HTMLAnchorElement;
    return { filename: anchor.download, text: await readBlob(createObjectURL.mock.calls.at(-1)![0]) };
  };

  it("downloads a CSV template with the columns the upload accepts", async () => {
    const user = await openDialog();

    await user.click(screen.getByRole("button", { name: /download csv template/i }));

    const { filename, text } = await lastDownload();
    expect(filename).toBe("bulk_users_template.csv");
    expect(text.split(/\r?\n/)[0]).toBe("user_email,user_role,teams,max_budget,budget_duration,models");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:download");
  });

  it("exposes the drop zone as a label for a keyboard-reachable file input", async () => {
    await openDialog();

    const input = fileInput();
    expect(input).toHaveAttribute("type", "file");
    expect(input).toHaveAttribute("accept", ".csv");
    expect(input).toBeVisible();
    expect(input.closest("label")).toHaveAttribute("for", input.id);
    const danglingLabels = [...document.querySelectorAll("label[for]")].filter(
      (label) => document.getElementById(label.getAttribute("for") as string) === null,
    );
    expect(danglingLabels).toEqual([]);
  });

  it("parses a CSV dropped onto the drop zone", async () => {
    await openDialog();

    fireEvent.drop(fileInput(), {
      dataTransfer: {
        files: [csvFile("user_email,user_role\nnew.hire@example.com,internal_user\n")],
        types: ["Files"],
      },
    });

    expect(await screen.findByText("new.hire@example.com")).toBeInTheDocument();
  });

  it("shows a structure error for a CSV missing required columns", async () => {
    await openDialog();

    fireEvent.change(fileInput(), { target: { files: [csvFile("email\nnew.hire@example.com\n")] } });

    expect(await screen.findByText("CSV Structure Error")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Your CSV is missing these required columns: user_email, user_role. Please add these columns to your CSV file.",
      ),
    ).toBeInTheDocument();
  });

  it("previews valid and invalid rows, creates the users, and downloads the results", async () => {
    const onUsersCreated = vi.fn();
    vi.spyOn(console, "error").mockImplementation(() => {});
    userCreateCall.mockImplementation(async (_token: string, _userId: null, body: { user_email: string }) => {
      const [name, domain] = body.user_email.split("@");
      if (!domain) throw new Error("invalid email");
      return { key: `sk-${name}`, user_id: `user-${name}` };
    });
    invitationCreateCall.mockImplementation(async (_token: string, userId: string) => ({ id: `inv-${userId}` }));
    const user = await openDialog(onUsersCreated);

    fireEvent.change(fileInput(), {
      target: {
        files: [
          csvFile(
            "user_email,user_role,teams,max_budget\n" +
              "first@example.com,internal_user,team-a,10\n" +
              "broken,nope,team-a,10\n" +
              "second@example.com,proxy_admin,,\n",
          ),
        ],
      },
    });

    expect(await screen.findByText("Review and create users")).toBeInTheDocument();
    expect(
      screen.getByText("Found 1 row(s) with errors out of 3 total rows. Please correct them before proceeding."),
    ).toBeInTheDocument();
    expect(screen.getByText("2 of 3 users valid")).toBeInTheDocument();
    const brokenRow = screen.getByRole("row", { name: /broken/ });
    expect(within(brokenRow).getByText("Invalid")).toBeInTheDocument();
    expect(within(brokenRow).getByText(/Invalid email format/)).toBeInTheDocument();

    await user.click(screen.getAllByRole("button", { name: "Create 2 Users" })[0]);

    expect(await screen.findByText("User Creation Results")).toBeInTheDocument();
    expect(await screen.findByText("2 Successful")).toBeInTheDocument();
    const firstRequest = {
      user_email: "first@example.com",
      user_role: "internal_user",
      teams: ["team-a"],
      max_budget: 10,
    };
    expect(userCreateCall).toHaveBeenCalledWith("test-token", null, firstRequest);
    expect(userCreateCall).toHaveBeenCalledWith("test-token", null, {
      user_email: "second@example.com",
      user_role: "proxy_admin",
    });
    expect(invitationCreateCall).toHaveBeenCalledWith("test-token", "user-first");
    expect(invitationCreateCall).toHaveBeenCalledWith("test-token", "user-second");
    expect(onUsersCreated).toHaveBeenCalledOnce();

    const firstRow = screen.getByRole("row", { name: /first@example.com/ });
    expect(within(firstRow).getByText("Success")).toBeInTheDocument();
    expect(
      within(firstRow).getByText(`${window.location.origin}/ui/onboarding?invitation_id=inv-user-first`),
    ).toBeInTheDocument();
    expect(within(firstRow).getByRole("button", { name: "Copy" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /download user credentials/i }));

    const { filename, text } = await lastDownload();
    expect(filename).toBe("bulk_users_results.csv");
    const lines = text.split(/\r?\n/);
    expect(lines[0]).toBe("user_email,user_role,status,key,invitation_link,error");
    expect(lines).toContain(
      `first@example.com,internal_user,success,sk-first,${window.location.origin}/ui/onboarding?invitation_id=inv-user-first,`,
    );
    expect(lines).toContain(
      `second@example.com,proxy_admin,success,sk-second,${window.location.origin}/ui/onboarding?invitation_id=inv-user-second,`,
    );
  });

  it("waits for each user to be created before creating the next", async () => {
    const pending: Array<(value: { key: string; user_id: string }) => void> = [];
    userCreateCall.mockImplementation(() => new Promise((resolve) => pending.push(resolve)));
    invitationCreateCall.mockResolvedValue({ id: "inv" });
    const user = await openDialog();
    fireEvent.change(fileInput(), {
      target: { files: [csvFile("user_email,user_role\na@example.com,internal_user\nb@example.com,internal_user")] },
    });

    await user.click((await screen.findAllByRole("button", { name: "Create 2 Users" }))[0]);

    expect(userCreateCall).toHaveBeenCalledOnce();
    pending[0]({ key: "sk-a", user_id: "user-a" });
    await vi.waitFor(() => expect(userCreateCall).toHaveBeenCalledTimes(2));
    pending[1]({ key: "sk-b", user_id: "user-b" });
    expect(await screen.findByText("2 Successful")).toBeInTheDocument();
  });

  it("pages the preview five rows at a time", async () => {
    const user = await openDialog();
    const lines = Array.from({ length: 7 }, (_, i) => `user${i + 1}@example.com,internal_user`);

    fireEvent.change(fileInput(), { target: { files: [csvFile(["user_email,user_role", ...lines].join("\n"))] } });

    expect(await screen.findByText("Page 1 of 2")).toBeInTheDocument();
    expect(screen.getByText("user5@example.com")).toBeInTheDocument();
    expect(screen.queryByText("user6@example.com")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Next" }));

    expect(screen.getByText("Page 2 of 2")).toBeInTheDocument();
    expect(screen.getByText("user7@example.com")).toBeInTheDocument();
    expect(screen.queryByText("user1@example.com")).not.toBeInTheDocument();
  });
});
