import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { toast } from "sonner";
import { fireEvent, renderWithProviders, screen, testQueryClient, waitFor } from "../../../../../tests/test-utils";
import CreateCustomerDialog from "./CreateCustomerDialog";

const creations: unknown[] = [];
let rejectCreation = false;
let freeTierReached = false;
const onCreated = vi.fn();

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

function Harness() {
  const [open, setOpen] = useState(true);
  return (
    <>
      <button onClick={() => setOpen(true)}>Open customer form</button>
      <CreateCustomerDialog open={open} onOpenChange={setOpen} onCreated={onCreated} />
    </>
  );
}

beforeEach(() => {
  creations.length = 0;
  rejectCreation = false;
  freeTierReached = false;
  vi.mocked(toast.error).mockClear();
  onCreated.mockClear();
  testQueryClient.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (request: Request) => {
      const url = new URL(request.url);
      if (url.pathname.endsWith("/.well-known/litellm-ui-config")) {
        return Response.json({ server_root_path: "", proxy_base_url: "", admin_ui_disabled: false });
      }
      if (url.pathname === "/management/v1/customer_organizations" && request.method === "POST") {
        creations.push(await request.json());
        if (freeTierReached)
          return Response.json(
            { detail: "Free customer portal limit reached. Upgrade to Enterprise." },
            { status: 403 },
          );
        return rejectCreation
          ? Response.json({ detail: "Customer entity ID already exists." }, { status: 409 })
          : Response.json({ customer_org_id: "created-org", alias: "Example customer" }, { status: 201 });
      }
      throw new Error(`Unexpected request ${request.method} ${url.pathname}`);
    }),
  );
});

afterEach(() => {
  testQueryClient.clear();
  vi.unstubAllGlobals();
});

describe("CreateCustomerDialog", () => {
  it("surfaces the free-tier 403 detail", async () => {
    freeTierReached = true;
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "Sixth customer" } });
    fireEvent.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith(
        expect.stringContaining("Free customer portal limit reached. Upgrade to Enterprise."),
      ),
    );
  });
  it("adds a custom group and selects a remaining default when that group is removed", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "Example customer" } });
    await user.click(screen.getByRole("combobox", { name: "Group 2 at the cap" }));
    await user.click(screen.getByRole("option", { name: "Pause, they can request more" }));
    await user.click(screen.getByRole("button", { name: "Add budget group" }));
    fireEvent.change(screen.getByLabelText("Group 4 name"), { target: { value: "custom_user" } });
    fireEvent.change(screen.getByLabelText("Group 4 yearly budget (USD)"), { target: { value: "300" } });
    await user.click(screen.getByRole("radio", { name: "Use custom_user as default" }));
    await user.click(screen.getByRole("button", { name: "Remove custom_user" }));
    expect(screen.getByRole("radio", { name: "Use power_user as default" })).toBeChecked();
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("created-org"));
    expect(creations).toHaveLength(1);
    expect(creations[0]).toMatchObject({
      default_budgets: [
        { name: "power_user", cap_policy: "pause_until_reset" },
        { name: "default_user", cap_policy: "pause_allow_increase" },
        { name: "low_user", cap_policy: "pause_until_reset" },
      ],
      default_budget_template: "power_user",
    });
  });

  it("shows duplicate-name validation without making a request", async () => {
    const user = userEvent.setup();
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "Example customer" } });
    fireEvent.change(screen.getByLabelText("Group 2 name"), { target: { value: " power_user " } });
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Use a unique name for each budget group.");
    expect(creations).toHaveLength(0);
  });

  it("preserves entered budgets after failure and clears the form after a successful retry", async () => {
    const user = userEvent.setup();
    rejectCreation = true;
    renderWithProviders(<Harness />);
    fireEvent.change(screen.getByLabelText("Customer name"), { target: { value: "Example customer" } });
    fireEvent.change(screen.getByLabelText("Customer organization id"), { target: { value: "existing-id" } });
    fireEvent.change(screen.getByLabelText("Group 2 yearly budget (USD)"), { target: { value: "450" } });
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(creations).toHaveLength(1));
    expect(await screen.findByRole("button", { name: "Create customer" })).toBeEnabled();
    expect(screen.getByLabelText("Group 2 yearly budget (USD)")).toHaveValue(450);
    expect(onCreated).not.toHaveBeenCalled();
    rejectCreation = false;
    fireEvent.change(screen.getByLabelText("Customer organization id"), { target: { value: "new-id" } });
    await user.click(screen.getByRole("button", { name: "Create customer" }));
    await waitFor(() => expect(onCreated).toHaveBeenCalledWith("created-org"));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(creations[1]).toMatchObject({
      customer_org_id: "new-id",
      default_budgets: expect.arrayContaining([
        expect.objectContaining({
          name: "default_user",
          budget_limits: expect.arrayContaining([{ budget_duration: "12mo", max_budget: 450 }]),
        }),
      ]),
    });
    await user.click(screen.getByRole("button", { name: "Open customer form" }));
    expect(screen.getByLabelText("Customer name")).toHaveValue("");
    expect(screen.getByLabelText("Group 2 yearly budget (USD)")).toHaveValue(500);
    expect(screen.getByRole("radio", { name: "Use default_user as default" })).toBeChecked();
  });
});
