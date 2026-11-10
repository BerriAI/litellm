import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import type { Plugin } from "@/components/claude_code_plugins/types";
import SkillHubDashboard from "./SkillHubDashboard";

const skills: Plugin[] = [
  { id: "alpha", name: "alpha", source: {}, category: "Writing", domain: "Support", enabled: false, download_count: 1 },
  {
    id: "beta",
    name: "beta",
    source: {},
    category: "Analysis",
    domain: "Engineering",
    enabled: true,
    download_count: 7,
  },
];

describe("Skills Hub sorting", () => {
  it.each(["Category", "Domain", "Status", "Downloads"])("sorts by %s through the table header", async (header) => {
    const user = userEvent.setup();
    renderWithProviders(<SkillHubDashboard skills={skills} isLoading={false} />);
    expect(screen.getAllByRole("row")[1]).toHaveTextContent("alpha");
    await user.click(screen.getByRole("button", { name: header, exact: true }));
    await waitFor(() => expect(screen.getAllByRole("row")[1]).toHaveTextContent("beta"));
    await user.click(screen.getByRole("button", { name: header, exact: true }));
    await waitFor(() => expect(screen.getAllByRole("row")[1]).toHaveTextContent("alpha"));
  });
});
