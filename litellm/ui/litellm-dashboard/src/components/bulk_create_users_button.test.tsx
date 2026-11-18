import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import BulkCreateUsersButton from "./bulk_create_users_button";

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

const csvFile = () =>
  new File(["user_email,user_role\nnew.hire@example.com,internal_user\n"], "users.csv", { type: "text/csv" });

const openUploadStep = async () => {
  const user = userEvent.setup();
  render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
  await user.click(screen.getByText("+ Bulk Invite Users"));
  return user;
};

describe("BulkCreateUsersButton", () => {
  it("should render", () => {
    render(<BulkCreateUsersButton accessToken="test-token" teams={[]} possibleUIRoles={null} />);
    expect(screen.getByText("+ Bulk Invite Users")).toBeInTheDocument();
  });

  it("downloads a CSV template with the columns the upload accepts", async () => {
    const createObjectURL = vi.fn((_blob: Blob) => "blob:template");
    const revokeObjectURL = vi.fn();
    const { createObjectURL: originalCreate, revokeObjectURL: originalRevoke } = URL;
    URL.createObjectURL = createObjectURL;
    URL.revokeObjectURL = revokeObjectURL;
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    const user = await openUploadStep();
    await user.click(screen.getByRole("button", { name: /download csv template/i }));

    expect(anchorClick).toHaveBeenCalledOnce();
    const anchor = anchorClick.mock.contexts[0] as HTMLAnchorElement;
    expect(anchor.download).toBe("bulk_users_template.csv");
    expect(anchor.href).toBe("blob:template");

    const csv = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result as string);
      reader.readAsText(createObjectURL.mock.calls[0][0]);
    });
    const [headerRow] = csv.split(/\r?\n/);
    expect(headerRow).toBe("user_email,user_role,teams,max_budget,budget_duration,models");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:template");

    anchorClick.mockRestore();
    URL.createObjectURL = originalCreate;
    URL.revokeObjectURL = originalRevoke;
  });

  it("parses a CSV chosen through the file input", async () => {
    await openUploadStep();

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, { target: { files: [csvFile()] } });

    expect(await screen.findByText("new.hire@example.com")).toBeInTheDocument();
  });

  it("parses a CSV dropped onto the drop zone", async () => {
    await openUploadStep();

    const dropZone = screen.getByLabelText(/drag and drop your csv file here/i).closest("label");
    fireEvent.drop(dropZone as HTMLLabelElement, { dataTransfer: { files: [csvFile()], types: ["Files"] } });

    expect(await screen.findByText("new.hire@example.com")).toBeInTheDocument();
  });

  it("exposes the drop zone as a label for a keyboard-reachable file input", async () => {
    await openUploadStep();

    const fileInput = screen.getByLabelText(/drag and drop your csv file here/i) as HTMLInputElement;
    expect(fileInput).toHaveAttribute("type", "file");
    expect(fileInput).toHaveAttribute("accept", ".csv");
    expect(fileInput).toBeVisible();

    const dropZone = fileInput.closest("label") as HTMLLabelElement;
    expect(fileInput.id).not.toBe("");
    expect(dropZone.htmlFor).toBe(fileInput.id);

    const danglingLabels = [...document.querySelectorAll("label[for]")].filter(
      (label) => document.getElementById(label.getAttribute("for") as string) === null,
    );
    expect(danglingLabels).toEqual([]);
  });
});
