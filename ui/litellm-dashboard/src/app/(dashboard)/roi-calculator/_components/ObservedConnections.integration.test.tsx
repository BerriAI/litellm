import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import ObservedConnections from "./ObservedConnections";
import type { ObservedSettings } from "./observedData";

const settings: ObservedSettings = {
  id: "initial-github",
  source_provider: "github",
  api_url: "https://api.github.com",
  repos: [],
  has_token: false,
  connection_type: "token",
  update_interval_minutes: 1440,
  ready: false,
};
const app = { configured: true, api_url: null, callback_url: null };
afterEach(() => vi.unstubAllGlobals());

describe("observed ROI connections", () => {
  it("identifies the saved connection when editing its host", async () => {
    const writes: unknown[] = [];
    const existing = { ...settings, id: "saved-github", has_token: true, repos: ["org/service"], ready: true };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string, init: RequestInit) => {
        const path = new URL(input, "http://localhost").pathname;
        if (path.endsWith("/apps")) return Response.json({ github: app, gitlab: app });
        if (path.endsWith("/repositories")) return Response.json({ repositories: [], has_more: false });
        if (path.endsWith("/settings")) {
          writes.push(JSON.parse(String(init.body)));
          return Response.json({ ...existing, id: "enterprise-github", api_url: "https://git.example.test/api/v3" });
        }
        throw new Error(path);
      }),
    );
    const user = userEvent.setup();
    render(
      <ObservedConnections
        accessToken="gateway-test-token"
        settings={{ ...existing, connections: [existing] }}
        onClose={vi.fn()}
        onSaved={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: "Edit GitHub api.github.com" }));
    await user.click(screen.getByRole("button", { name: "Change connection" }));
    await user.click(screen.getByText("Self-hosted instance"));
    fireEvent.change(screen.getByLabelText("API URL"), { target: { value: "https://git.example.test/api/v3" } });
    fireEvent.change(screen.getByLabelText("GitHub access token"), { target: { value: "enterprise-test-token" } });
    await user.click(screen.getByRole("button", { name: "Continue" }));
    expect(await screen.findByRole("heading", { name: "Choose repositories" })).toBeInTheDocument();
    expect(writes).toEqual([
      {
        connection_id: "saved-github",
        source_provider: "github",
        api_url: "https://git.example.test/api/v3",
        token: "enterprise-test-token",
        repos: [],
        update_interval_minutes: 1440,
      },
    ]);
  });
  it("starts a GitHub installation when switching from a GitLab app connection", async () => {
    const starts = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string, init: RequestInit) => {
        const url = new URL(input, "http://localhost");
        if (url.pathname.endsWith("/apps"))
          return Response.json({ github: { ...app, can_install: true }, gitlab: app });
        if (url.pathname.endsWith("/repositories")) return Response.json({ repositories: [], has_more: false });
        starts(url.searchParams.get("install"), init.credentials);
        return Response.json({ detail: "Authorization test stopped before redirect" }, { status: 502 });
      }),
    );
    const user = userEvent.setup();
    const connected: ObservedSettings = {
      ...settings,
      source_provider: "gitlab",
      connection_type: "app",
      has_token: true,
    };
    render(<ObservedConnections accessToken="test-token" settings={connected} onClose={vi.fn()} onSaved={vi.fn()} />);
    await user.click(screen.getByRole("button", { name: "Change connection" }));
    await user.click(screen.getByRole("button", { name: "GitHub", exact: true }));
    const connect = screen.getByRole("button", { name: "Connect GitHub", exact: true });
    await waitFor(() => expect(connect).toBeEnabled());
    await user.click(connect);
    expect(await screen.findByRole("alert")).toHaveTextContent("Authorization test stopped before redirect");
    expect(starts).toHaveBeenCalledWith("true", "include");
  });
  it.each(["GitHub", "GitLab"] as const)(
    "connects %s with a token, saves repositories, and starts a sync",
    async (label) => {
      const provider = label === "GitHub" ? "github" : "gitlab";
      const apiUrl = provider === "github" ? "https://api.github.com" : "https://gitlab.com/api/v4";
      const saved = vi.fn();
      const closed = vi.fn();
      const writes: { path: string; body: unknown }[] = [];
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string, init: RequestInit) => {
          const path = new URL(input, "http://localhost").pathname;
          if (init.method === "PUT" || init.method === "POST")
            writes.push({ path, body: init.body ? JSON.parse(String(init.body)) : undefined });
          if (path.endsWith("/apps")) return Response.json({ github: app, gitlab: app });
          if (path.endsWith("/repositories"))
            return Response.json({
              repositories: [{ name: "org/service", visibility: "private", archived: false }],
              has_more: false,
            });
          if (path.endsWith("/settings")) {
            const request = JSON.parse(String(init.body)) as { repos: string[] };
            const connected = {
              ...settings,
              id: `saved-${provider}`,
              source_provider: provider,
              api_url: apiUrl,
              has_token: true,
              repos: request.repos,
              ready: request.repos.length > 0,
            };
            return Response.json(connected);
          }
          if (path.endsWith("/sync")) return Response.json({ running: true }, { status: 202 });
          throw new Error(path);
        }),
      );
      const user = userEvent.setup();
      render(
        <ObservedConnections accessToken="gateway-test-token" settings={settings} onClose={closed} onSaved={saved} />,
      );
      await user.click(screen.getByRole("button", { name: label, exact: true }));
      await user.click(screen.getByRole("button", { name: "Access token", exact: true }));
      fireEvent.change(screen.getByLabelText(`${label} access token`), { target: { value: "source-test-token" } });
      await user.click(screen.getByRole("button", { name: "Continue" }));
      await user.click(await screen.findByRole("checkbox", { name: /org\/service/ }));
      await user.click(screen.getByRole("button", { name: "Save and sync" }));
      await waitFor(() => expect(saved).toHaveBeenCalledOnce());
      expect(closed).toHaveBeenCalledOnce();
      expect(writes).toEqual([
        {
          path: "/roi-calculator/observed/settings",
          body: {
            source_provider: provider,
            api_url: apiUrl,
            token: "source-test-token",
            repos: [],
            update_interval_minutes: 1440,
          },
        },
        {
          path: "/roi-calculator/observed/settings",
          body: {
            connection_id: `saved-${provider}`,
            source_provider: provider,
            api_url: apiUrl,
            repos: ["org/service"],
            update_interval_minutes: 1440,
          },
        },
        { path: "/roi-calculator/observed/sync", body: undefined },
      ]);
    },
  );
  it.each(["GitHub", "GitLab"] as const)(
    "starts %s app authorization with a browser cookie and displays provider failures",
    async (label) => {
      const provider = label.toLowerCase();
      vi.stubGlobal(
        "fetch",
        vi.fn(async (input: string, init: RequestInit) => {
          if (String(input).endsWith("/apps")) return Response.json({ github: app, gitlab: app });
          expect(String(input)).toContain(`/oauth/${provider}/start`);
          expect(init.credentials).toBe("include");
          expect(init.method).toBe("POST");
          return Response.json({ detail: "Provider unavailable. Try again" }, { status: 502 });
        }),
      );
      const user = userEvent.setup();
      render(
        <ObservedConnections
          accessToken="gateway-test-token"
          settings={settings}
          onClose={vi.fn()}
          onSaved={vi.fn()}
        />,
      );
      await user.click(screen.getByRole("button", { name: label, exact: true }));
      const button = await screen.findByRole("button", { name: `Connect ${label}`, exact: true });
      await waitFor(() => expect(button).toBeEnabled());
      await user.click(button);
      expect(await screen.findByRole("alert")).toHaveTextContent("Provider unavailable. Try again");
      expect(button).toBeEnabled();
    },
  );
});
