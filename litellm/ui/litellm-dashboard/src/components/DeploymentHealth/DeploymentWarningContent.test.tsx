import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { deploymentWarningContent } from "./DeploymentWarningContent";

const NOW = new Date("2026-10-14T12:00:00Z");

describe("deploymentWarningContent", () => {
  it.each([
    { expiry: "2026-10-14", title: "Your LiteLLM Enterprise license expires today (Oct 14, 2026)" },
    { expiry: "2026-10-15", title: "Your LiteLLM Enterprise license expires in 1 day (Oct 15, 2026)" },
    { expiry: "2026-11-13", title: "Your LiteLLM Enterprise license expires in 30 days (Nov 13, 2026)" },
    { expiry: "2026-10-13", title: "Your LiteLLM Enterprise license expired on Oct 13, 2026" },
  ])("should title a license expiring on $expiry by its distance from today", ({ expiry, title }) => {
    expect(deploymentWarningContent(`license_expiry:${expiry}`, NOW).title).toBe(title);
  });

  it.each(["2026-10-20", "2026-10-01"])("should point a license expiring %s at sales", (expiry) => {
    render(<p>{deploymentWarningContent(`license_expiry:${expiry}`, NOW).fix}</p>);

    expect(screen.getByRole("link", { name: "sales@berri.ai" })).toHaveAttribute("href", "mailto:sales@berri.ai");
  });

  it("should tell admins which setting turns env credential login off", () => {
    render(<p>{deploymentWarningContent("env_credential_login", NOW).fix}</p>);

    expect(screen.getByText("general_settings.disable_env_credential_login: true")).toBeInTheDocument();
  });

  it.each(["env_credential_login", "no_redis", "detailed_debug", "license_expiry:2026-10-20"])(
    "should link %s to the LiteLLM docs",
    (warningId) => {
      expect(deploymentWarningContent(warningId, NOW).docsUrl).toMatch(/^https:\/\/docs\.litellm\.ai\/docs\//);
    },
  );

  it("should fall back to the raw id for a warning the dashboard does not know yet", () => {
    expect(deploymentWarningContent("future_warning", NOW)).toEqual({ title: "future_warning" });
  });
});
