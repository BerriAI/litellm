import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DataTable } from "@/components/shared/DataTable";
import { Plugin } from "@/components/claude_code_plugins/types";
import { getSkillHubTableColumns } from "./SkillHubTableColumns";

const mockSkill: Plugin = {
  id: "skill-1",
  name: "pdf-tools",
  description: "Work with PDF files",
  source: { source: "github", repo: "org/pdf-tools" },
  category: "documents",
  domain: "Productivity",
  enabled: true,
};

function renderTable(data: Plugin[], onSkillClick = vi.fn()) {
  render(
    <DataTable
      data={data}
      columns={getSkillHubTableColumns({ onSkillClick })}
      getRowId={(skill, index) => skill.id || String(index)}
      sortingMode="client"
      size="compact"
    />,
  );
  return onSkillClick;
}

describe("getSkillHubTableColumns", () => {
  it("renders the skill row with category and domain", () => {
    renderTable([mockSkill]);
    expect(screen.getByText("pdf-tools")).toBeInTheDocument();
    expect(screen.getByText("documents")).toBeInTheDocument();
    expect(screen.getByText("Productivity")).toBeInTheDocument();
  });

  it("links to the github source", () => {
    renderTable([mockSkill]);
    const link = screen.getByRole("link", { name: /org\/pdf-tools/ });
    expect(link).toHaveAttribute("href", "https://github.com/org/pdf-tools");
  });

  it("shows Public for enabled skills and Draft for disabled ones", () => {
    renderTable([mockSkill, { ...mockSkill, id: "skill-2", name: "draft-skill", enabled: false }]);
    expect(screen.getByText("Public")).toBeInTheDocument();
    expect(screen.getByText("Draft")).toBeInTheDocument();
  });

  it("opens the skill detail when the name is clicked", async () => {
    const user = userEvent.setup();
    const onSkillClick = renderTable([mockSkill]);
    await user.click(screen.getByRole("button", { name: "pdf-tools" }));
    expect(onSkillClick).toHaveBeenCalledWith(mockSkill);
  });

  it("opens the skill detail from the actions menu", async () => {
    const user = userEvent.setup();
    const onSkillClick = renderTable([mockSkill]);
    await user.click(screen.getByTestId("skill-hub-actions-skill-1"));
    await user.click(await screen.findByTestId("skill-hub-action-details"));
    expect(onSkillClick).toHaveBeenCalledWith(mockSkill);
  });

  it("copies the skill name from the actions menu", async () => {
    const user = userEvent.setup();
    renderTable([mockSkill]);
    await user.click(screen.getByTestId("skill-hub-actions-skill-1"));
    await user.click(await screen.findByTestId("skill-hub-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("pdf-tools");
  });
});
