import { useState } from "react";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { components } from "@/lib/http/schema";
import { toast } from "@/lib/toast";
import { jsonResponse, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { LensControls } from "./LensControls";
import type { LensPolicy } from "./lensPolicy";

const service: components["schemas"]["ServiceConnection"] = {
  browser_url: "",
  configured: true,
  connected: true,
  release: "test",
  url: "https://lens.example.com",
  status: { credentials_ready: true, protocol_version: 7, public_contract: 1, release: "test", storage_ready: true },
};

function EditableKey({ onChange }: { onChange: (policy: LensPolicy) => Promise<void> }) {
  const [policy, setPolicy] = useState<LensPolicy>(null);
  return (
    <LensControls
      kind="key"
      policy={policy}
      teamPolicy={false}
      canEdit
      href="/ui/lens?tab=traces&api_key_hash=key-hash"
      onPolicyChange={async (next) => {
        await onChange(next);
        setPolicy(next);
      }}
    />
  );
}

beforeEach(() => {
  testQueryClient.clear();
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse(service)));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("Lens controls", () => {
  it.each(["team", "key"] as const)(
    "shows an unknown inherited %s policy when the global status is unavailable",
    async (kind) => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "Unavailable" }, 503)));
      const onChange = vi.fn<(policy: LensPolicy) => Promise<void>>();
      renderWithProviders(<LensControls kind={kind} policy={null} canEdit href="/ui/lens" onPolicyChange={onChange} />);
      expect(await screen.findByText("Unknown")).toBeInTheDocument();
      expect(screen.queryByText("Off", { exact: true })).not.toBeInTheDocument();
      expect(screen.getByText("Lens connection status is unavailable")).toBeInTheDocument();
      const toggle = screen.getByRole("switch", { name: "Lens" });
      expect(toggle).toHaveAttribute("aria-disabled", "true");
      fireEvent.click(toggle);
      expect(onChange).not.toHaveBeenCalled();
      expect(screen.getByRole("link", { name: "Open Lens" })).toBeInTheDocument();
    },
  );

  it.each([
    ["team", true],
    ["team", false],
    ["key", true],
    ["key", false],
  ] as const)("keeps an explicit %s policy of %s visible during a connection-status error", async (kind, policy) => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "Unavailable" }, 503)));
    renderWithProviders(<LensControls kind={kind} policy={policy} canEdit href={null} onPolicyChange={vi.fn()} />);
    await screen.findByText("Lens connection status is unavailable");
    expect(screen.getByText(policy ? "On" : "Off", { exact: true })).toBeInTheDocument();
    expect(screen.queryByText("Unknown")).not.toBeInTheDocument();
  });

  it.each([true, false])(
    "keeps the known inherited team setting of %s during a connection-status error",
    async (teamPolicy) => {
      vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ error: "Unavailable" }, 503)));
      renderWithProviders(
        <LensControls kind="key" policy={null} teamPolicy={teamPolicy} canEdit href={null} onPolicyChange={vi.fn()} />,
      );
      await screen.findByText("Lens connection status is unavailable");
      expect(screen.getByText(teamPolicy ? "On" : "Off", { exact: true })).toBeInTheDocument();
      expect(screen.queryByText("Unknown")).not.toBeInTheDocument();
    },
  );

  it("shows an unknown inherited key policy when the team lookup fails", async () => {
    renderWithProviders(
      <LensControls kind="key" policy={null} teamPolicyStatus="error" canEdit href={null} onPolicyChange={vi.fn()} />,
    );
    const toggle = screen.getByRole("switch", { name: "Lens" });
    await waitFor(() => expect(toggle).toHaveAttribute("aria-busy", "false"));
    expect(screen.getByText("Unknown")).toBeInTheDocument();
    expect(screen.queryByText("On", { exact: true })).not.toBeInTheDocument();
    expect(toggle).toHaveAttribute("aria-disabled", "true");
  });

  it("lets a key override a disabled team and reset to its team setting", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn<(policy: LensPolicy) => Promise<void>>().mockResolvedValue(undefined);
    renderWithProviders(<EditableKey onChange={onChange} />);
    const toggle = screen.getByRole("switch", { name: "Lens" });
    await waitFor(() => expect(toggle).not.toHaveAttribute("aria-disabled", "true"));
    expect(toggle).not.toBeChecked();
    expect(screen.getByText("Uses team setting")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Use team setting" })).not.toBeInTheDocument();

    await user.click(toggle);
    expect(onChange).toHaveBeenCalledWith(true);
    expect(toggle).toBeChecked();
    expect(screen.getByText("Custom setting")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Use team setting" }));
    expect(onChange).toHaveBeenLastCalledWith(null);
    expect(toggle).not.toBeChecked();
    expect(screen.getByText("Uses team setting")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
      "href",
      "/ui/lens?tab=traces&api_key_hash=key-hash",
    );
  });

  it("keeps the saved state when persistence fails", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn<(policy: LensPolicy) => Promise<void>>().mockRejectedValue(new Error("Denied"));
    renderWithProviders(<EditableKey onChange={onChange} />);
    const toggle = screen.getByRole("switch", { name: "Lens" });
    await waitFor(() => expect(toggle).not.toHaveAttribute("aria-disabled", "true"));
    await user.click(toggle);
    await waitFor(() => expect(toast.fromError).toHaveBeenCalledWith("Lens logging setting could not be saved"));
    expect(toggle).not.toBeChecked();
    expect(toggle).not.toHaveAttribute("aria-disabled", "true");
  });

  it("waits for the inherited team policy before allowing a key override", async () => {
    const onChange = vi.fn<(policy: LensPolicy) => Promise<void>>();
    const controls = {
      kind: "key" as const,
      policy: null,
      canEdit: true,
      href: "/ui/lens?tab=traces&api_key_hash=key-hash",
      onPolicyChange: onChange,
    };
    const { rerender } = renderWithProviders(<LensControls {...controls} teamPolicyStatus="loading" />);
    const toggle = screen.getByRole("switch", { name: "Lens" });
    expect(toggle).toHaveAttribute("aria-disabled", "true");
    expect(toggle).toHaveAttribute("aria-busy", "true");
    fireEvent.click(toggle);
    expect(onChange).not.toHaveBeenCalled();

    rerender(<LensControls {...controls} teamPolicy={false} teamPolicyStatus="ready" />);
    await waitFor(() => expect(toggle).not.toHaveAttribute("aria-disabled", "true"));
    expect(toggle).not.toBeChecked();
    expect(toggle).toHaveAttribute("aria-busy", "false");
  });

  it("does not open an unfiltered dashboard when the entity link is unavailable", () => {
    renderWithProviders(<LensControls kind="key" policy={false} canEdit href={null} onPolicyChange={vi.fn()} />);
    expect(screen.getByRole("button", { name: "Open Lens" })).toBeDisabled();
    expect(screen.queryByRole("link", { name: "Open Lens" })).not.toBeInTheDocument();
  });

  it("keeps historical traces reachable when a read-only user has logging off", async () => {
    const onChange = vi.fn<(policy: LensPolicy) => Promise<void>>();
    renderWithProviders(
      <LensControls
        kind="team"
        policy={false}
        canEdit={false}
        href="/ui/lens?tab=traces&team_id=team-id"
        onPolicyChange={onChange}
      />,
    );
    const toggle = screen.getByRole("switch", { name: "Lens" });
    expect(toggle).toHaveAttribute("aria-disabled", "true");
    expect(toggle).not.toBeChecked();
    fireEvent.click(toggle);
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.getByRole("link", { name: "Open Lens" })).toHaveAttribute(
      "href",
      "/ui/lens?tab=traces&team_id=team-id",
    );
  });
});
