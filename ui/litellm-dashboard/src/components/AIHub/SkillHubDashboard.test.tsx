import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "../../../tests/test-utils";
import SkillHubDashboard from "./SkillHubDashboard";
import { Plugin } from "@/components/claude_code_plugins/types";

const SKILLS: Plugin[] = [
  {
    id: "skill-1",
    name: "deploy-helper",
    description: "Helps with deployment runbooks",
    source: { source: "github", repo: "example/deploy-helper" },
    domain: "DevOps",
    namespace: "platform",
    enabled: true,
  },
  {
    id: "skill-2",
    name: "support-triage",
    description: "Helps triage support requests",
    source: { source: "github", repo: "example/support-triage" },
    domain: "Support",
    namespace: "support",
    enabled: true,
  },
  {
    id: "skill-3",
    name: "release-notes",
    description: "Drafts release notes",
    source: { source: "github", repo: "example/release-notes" },
    domain: "DevOps",
    namespace: "platform",
    enabled: true,
  },
];

describe("SkillHubDashboard", () => {
  it("should filter skills by namespace", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SkillHubDashboard skills={SKILLS} isLoading={false} />);

    expect(screen.getByText("deploy-helper")).toBeInTheDocument();
    expect(screen.getByText("support-triage")).toBeInTheDocument();
    expect(screen.getByText("release-notes")).toBeInTheDocument();
    expect(screen.getByText("Namespace")).toBeInTheDocument();

    const namespaceSelect = screen.getByText("All Namespaces").closest(".ant-select") as HTMLElement;
    await user.click(namespaceSelect.querySelector(".ant-select-selector")!);

    await waitFor(() => {
      expect(document.querySelector('[title="support"].ant-select-item-option')).toBeInTheDocument();
    });

    await user.click(document.querySelector('[title="support"].ant-select-item-option')!);

    await waitFor(() => {
      expect(screen.queryByText("deploy-helper")).not.toBeInTheDocument();
      expect(screen.getByText("support-triage")).toBeInTheDocument();
      expect(screen.queryByText("release-notes")).not.toBeInTheDocument();
      expect(screen.getByText("Showing 1 of 3 skills")).toBeInTheDocument();
    });
  });
});
