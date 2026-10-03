import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import ObservedAccounts from "./ObservedAccounts";

afterEach(() => vi.unstubAllGlobals());
it("links several usernames to one internal email in a single save", async () => {
  const writes: unknown[] = [];
  const saved = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_input: string, init: RequestInit) => {
      if (init.method === "PUT") {
        writes.push(JSON.parse(String(init.body)));
        return Response.json({ report: null });
      }
      return Response.json({
        gateway_emails: ["ari@example.test"],
        identity_map: { old: "ari@example.test" },
        unmatched_logins: ["new"],
      });
    }),
  );
  const user = userEvent.setup();
  render(<ObservedAccounts accessToken="gateway-test-token" people={[]} onClose={vi.fn()} onSaved={saved} />);
  await screen.findByText("1 unmatched accounts");
  fireEvent.change(screen.getByLabelText("Internal email"), { target: { value: "ari@example.test" } });
  expect(screen.getByLabelText("Source usernames")).toHaveValue("old");
  fireEvent.change(screen.getByLabelText("Source usernames"), { target: { value: "@Old, new, NEW" } });
  await user.click(screen.getByRole("button", { name: "Save accounts" }));
  await waitFor(() => expect(saved).toHaveBeenCalledOnce());
  expect(writes).toEqual([{ email: "ari@example.test", logins: ["old", "new"] }]);
});
it("keeps a conflicting link editable", async () => {
  const saved = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (_input: string, init: RequestInit) =>
      init.method === "PUT"
        ? Response.json({ detail: "An account is already linked to another email. Unlink it first" }, { status: 409 })
        : Response.json({ gateway_emails: ["ari@example.test"], identity_map: {}, unmatched_logins: [] }),
    ),
  );
  const user = userEvent.setup();
  render(
    <ObservedAccounts
      accessToken="gateway-test-token"
      people={[]}
      initialEmail="ari@example.test"
      onClose={vi.fn()}
      onSaved={saved}
    />,
  );
  fireEvent.change(screen.getByLabelText("Source usernames"), { target: { value: "old, new" } });
  const button = screen.getByRole("button", { name: "Save accounts" });
  await waitFor(() => expect(button).toBeEnabled());
  await user.click(button);
  expect(await screen.findByRole("alert")).toHaveTextContent("An account is already linked to another email");
  expect(screen.getByLabelText("Source usernames")).toHaveValue("old, new");
  expect(saved).not.toHaveBeenCalled();
});
