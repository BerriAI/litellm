import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";

import { fireEvent, renderWithProviders, screen, testQueryClient, within } from "../../../../../tests/test-utils";
import { SecretsVaultView } from "./SecretsVaultView";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

const STAMP = "2026-01-01T00:00:00Z";

type Stored = { value: string; description: string | null };
const vault = new Map<string, Stored>();
const requests: string[] = [];

const metadata = (name: string, stored: Stored) => ({
  secret_name: name,
  description: stored.description,
  created_at: STAMP,
  updated_at: STAMP,
  last_accessed_at: null,
});

beforeEach(() => {
  vault.clear();
  requests.length = 0;
  vault.set("GITHUB_TOKEN", { value: "ghp_plaintext", description: "CI token" });
  testQueryClient.clear();
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof fetch>(async (input, init) => {
      const request = new Request(input, init);
      const { pathname } = new URL(request.url);
      if (!pathname.startsWith("/vault/secrets")) return Response.json({});
      requests.push(`${request.method} ${pathname}`);
      if (pathname === "/vault/secrets") {
        return Response.json({ secrets: [...vault].map(([name, stored]) => metadata(name, stored)) });
      }
      const name = decodeURIComponent(pathname.slice("/vault/secrets/".length));
      if (request.method === "PUT") {
        const stored: Stored = await request.json();
        vault.set(name, stored);
        return Response.json(metadata(name, stored));
      }
      if (request.method === "DELETE") {
        return Response.json({ secret_name: name, deleted: vault.delete(name) });
      }
      const stored = vault.get(name);
      return stored
        ? Response.json({ ...metadata(name, stored), value: stored.value })
        : Response.json({ error: { message: "not found" } }, { status: 404 });
    }),
  );
});

describe("SecretsVaultView", () => {
  it("lists secrets without fetching or showing their values", async () => {
    renderWithProviders(<SecretsVaultView />);

    expect(await screen.findByText("GITHUB_TOKEN")).toBeInTheDocument();
    expect(screen.getByText("CI token")).toBeInTheDocument();
    expect(screen.queryByText("ghp_plaintext")).not.toBeInTheDocument();
    expect(requests).toEqual(["GET /vault/secrets"]);
  });

  it("reveals a value on request and hides it again", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SecretsVaultView />);

    await user.click(await screen.findByRole("button", { name: "Reveal GITHUB_TOKEN" }));
    expect(await screen.findByText("ghp_plaintext")).toBeInTheDocument();
    expect(requests).toContain("GET /vault/secrets/GITHUB_TOKEN");

    await user.click(screen.getByRole("button", { name: "Hide GITHUB_TOKEN" }));
    expect(screen.queryByText("ghp_plaintext")).not.toBeInTheDocument();
  });

  it("creates a secret under the name the user typed", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SecretsVaultView />);

    await user.click(await screen.findByRole("button", { name: "New secret" }));
    const dialog = await screen.findByRole("dialog");
    fireEvent.change(within(dialog).getByLabelText("Name"), { target: { value: "NPM_TOKEN" } });
    fireEvent.change(within(dialog).getByLabelText("Value"), { target: { value: "npm_secret" } });
    await user.click(within(dialog).getByRole("button", { name: "Create secret" }));

    expect(await screen.findByText("NPM_TOKEN")).toBeInTheDocument();
    expect(vault.get("NPM_TOKEN")).toEqual({ value: "npm_secret", description: null });
    expect(screen.queryByText("npm_secret")).not.toBeInTheDocument();
  });

  it("deletes a secret after confirmation", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SecretsVaultView />);

    await user.click(await screen.findByRole("button", { name: "Delete GITHUB_TOKEN" }));
    const dialog = await screen.findByRole("dialog");
    await user.click(within(dialog).getByRole("button", { name: /delete/i }));

    expect(await screen.findByText("No secrets yet")).toBeInTheDocument();
    expect(vault.has("GITHUB_TOKEN")).toBe(false);
  });
});
