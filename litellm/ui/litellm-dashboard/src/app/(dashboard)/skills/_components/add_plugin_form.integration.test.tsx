import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import AddPluginForm from "./add_plugin_form";
import { importSkillHubSkill, uploadSkillHubSkill } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  importSkillHubSkill: vi.fn().mockResolvedValue({ id: "skill" }),
  uploadSkillHubSkill: vi.fn().mockResolvedValue({ id: "skill" }),
}));
const importSkill = vi.mocked(importSkillHubSkill);
const uploadSkill = vi.mocked(uploadSkillHubSkill);
const props = { visible: true, onClose: vi.fn(), accessToken: "sk-test", onSuccess: vi.fn() };
const file = new File(["skill archive"], "personal-skill.zip", { type: "application/zip" });
const chooseFile = (selected = file) =>
  fireEvent.change(screen.getByLabelText("Skill ZIP file"), { target: { files: [selected] } });
const change = (label: string, value: string) => fireEvent.change(screen.getByLabelText(label), { target: { value } });
const submit = () => fireEvent.click(screen.getByRole("button", { name: "Register skill" }));
const openImport = (url = "https://github.com/example/skills") => {
  renderWithProviders(<AddPluginForm {...props} />);
  fireEvent.click(screen.getByRole("radio", { name: "Add from URL" }));
  change("Source URL", url);
  if (!url.startsWith("https://skills.sh/")) {
    fireEvent.click(screen.getByRole("radio", { name: "Store a copy in LiteLLM" }));
  }
};

describe("AddPluginForm", () => {
  beforeEach(() => vi.clearAllMocks());

  it("offers ZIP or URL, with managed copies available for supported URL hosts", () => {
    renderWithProviders(<AddPluginForm {...props} />);
    expect(screen.getByRole("radio", { name: "Upload ZIP" })).toBeChecked();
    expect(screen.getByText(/frontmatter supplies the skill name and description/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Add from URL" }));
    expect(screen.getByLabelText("Source URL")).toBeInTheDocument();
    expect(screen.getByLabelText("Path (optional)")).toBeEnabled();
    expect(screen.getByRole("radio", { name: "Use the external source" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Store a copy in LiteLLM" })).toHaveAttribute("aria-disabled", "true");
    change("Source URL", "  HTTPS://www.GitHub.com/example/skills  ");
    fireEvent.click(screen.getByRole("radio", { name: "Store a copy in LiteLLM" }));
    expect(screen.getByLabelText("Registry name (optional)")).toHaveValue("");
    expect(screen.getByRole("radio", { name: "Add from URL" })).toBeChecked();
    change("Skill selector (optional)", "previous-draft");
    change("Source URL", " www.Skills.sh/example/skills/review ");
    expect(screen.getByLabelText("Skill selector (optional)")).toBeDisabled();
    expect(screen.getByLabelText("Skill selector (optional)")).toHaveValue("review");
    expect(screen.getByRole("radio", { name: "Use the external source" })).toHaveAttribute("aria-disabled", "true");
    fireEvent.click(screen.getByRole("radio", { name: "Upload ZIP" }));
    fireEvent.click(screen.getByRole("radio", { name: "Add from URL" }));
    expect(screen.getByRole("radio", { name: "Store a copy in LiteLLM" })).toBeChecked();
    change("Source URL", "GitHub.com/example/skills");
    expect(screen.getByLabelText("Skill selector (optional)")).toBeEnabled();
    expect(screen.getByLabelText("Skill selector (optional)")).toHaveValue("previous-draft");
  });

  it.each(["https://github.com/anthropics/claude-code", "https://skills.sh/example/skills/review"])(
    "submits the canonical source URL %s",
    async (url) => {
      openImport();
      change(
        "Source URL",
        ` ${url.replace("https://", "HTTPS://www.").replace(".com", ".COM").replace(".sh", ".SH")} `,
      );
      submit();
      await waitFor(() => expect(importSkill).toHaveBeenCalledWith("sk-test", { url }));
      expect(props.onSuccess).toHaveBeenCalledOnce();
    },
  );

  it.each(["tree/release/v2/skills/review", "blob/release/v2/skills/review/SKILL.md"])(
    "imports one canonical selection from %s after changing sources",
    async (location) => {
      openImport();
      change("Skill selector (optional)", "review");
      change("Git ref", "release/v2");
      const url = "https://github.com/example/other-skills";
      change("Source URL", `${url}/tree/release/v2`);
      expect(screen.getByLabelText("Path (optional)")).toBeEnabled();
      change("Path (optional)", "previous-draft");
      change("Version (Optional)", "3.0.0");
      change("Source URL", `${url}/${location}`);
      expect(screen.getByLabelText("Path (optional)")).toBeDisabled();
      expect(screen.getByLabelText("Path (optional)")).toHaveValue("skills/review");
      expect(screen.getByLabelText("Skill selector (optional)")).toHaveAccessibleDescription(/both must match/);
      submit();
      const expected = { url, skill: "review", ref: "release/v2", path: "skills/review", version: "3.0.0" };
      await waitFor(() => expect(importSkill).toHaveBeenCalledWith("sk-test", expected));
    },
  );

  it("switches unsupported import hosts to references and preserves source drafts", async () => {
    openImport();
    change("Registry name (optional)", "review");
    const repository = "https://gitlab.com/team/repo";
    const archive = "https://artifacts.example.test/skill.zip";
    change("Source URL", repository);
    expect(screen.getByRole("radio", { name: "Use the external source" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Store a copy in LiteLLM" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.queryByLabelText("Skill selector (optional)")).not.toBeInTheDocument();
    change("Path (optional)", "plugins/review");
    change("Source URL", archive);
    expect(screen.getByLabelText("Path (optional)")).toBeDisabled();
    expect(screen.getByLabelText("Path (optional)")).toHaveValue("");
    expect(screen.queryByLabelText("Git ref")).not.toBeInTheDocument();
    change("Source URL", repository);
    expect(screen.getByLabelText("Path (optional)")).toBeEnabled();
    expect(screen.getByLabelText("Path (optional)")).toHaveValue("plugins/review");
    change("Git ref", "release/v2");
    change("Source URL", archive);
    submit();
    await waitFor(() =>
      expect(importSkill).toHaveBeenCalledWith("sk-test", {
        name: "review",
        source: { source: "archive", url: archive },
      }),
    );
  });

  it("shows the GitLab browser directory and sends its clone URL, full ref and path", async () => {
    renderWithProviders(<AddPluginForm {...props} />);
    fireEvent.click(screen.getByRole("radio", { name: "Add from URL" }));
    change("Registry name", "review");
    change("Source URL", "https://gitlab.com/team/repo");
    change("Path (optional)", "previous-draft");
    change("Source URL", "https://gitlab.com/team/subgroup/repo/-/blob/release/v2/plugins/review/SKILL.md");
    expect(screen.getByLabelText("Git ref")).not.toHaveAttribute("placeholder");
    expect(screen.getByLabelText("Git ref")).toHaveAccessibleDescription(/Enter the full Git ref from this URL/);
    expect(screen.getByLabelText("Path (optional)")).toBeDisabled();
    expect(screen.getByLabelText("Path (optional)")).toHaveValue("");
    expect(screen.queryByText(/^Detected:/)).not.toBeInTheDocument();
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Enter the full Git ref from this URL");
    expect(importSkill).not.toHaveBeenCalled();
    change("Git ref", "release/v2");
    expect(screen.getByLabelText("Path (optional)")).toBeDisabled();
    expect(screen.getByLabelText("Path (optional)")).toHaveValue("plugins/review");
    expect(screen.getByText(/^Detected:/)).toHaveTextContent("plugins/review (release/v2)");
    submit();
    const expected = {
      name: "review",
      source: {
        source: "git-subdir",
        url: "https://gitlab.com/team/subgroup/repo",
        ref: "release/v2",
        path: "plugins/review",
      },
    };
    await waitFor(() => expect(importSkill).toHaveBeenCalledWith("sk-test", expected));
  });

  it.each([
    "Unable to import skill",
    "A skill named 'review' already exists. Update the existing skill instead of adding it again.",
  ])("shows the API failure without closing the form: %s", async (message) => {
    importSkill.mockRejectedValueOnce(new Error(message));
    openImport();
    change("Source URL", "https://github.com/example/skills");
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(props.onSuccess).not.toHaveBeenCalled();
    expect(props.onClose).not.toHaveBeenCalled();
  });

  it("preserves the file and explicit version while a pending or failed upload is retried", async () => {
    const pending = Promise.withResolvers<never>();
    uploadSkill.mockReturnValueOnce(pending.promise);
    renderWithProviders(<AddPluginForm {...props} />);
    chooseFile();
    change("Version (Optional)", "2.0.0");
    submit();
    expect(await screen.findByRole("button", { name: "Saving..." })).toBeDisabled();
    expect(props.onClose).not.toHaveBeenCalled();
    await act(async () => pending.reject(new Error("Unable to save skill")));
    expect(await screen.findByRole("alert")).toHaveTextContent("Unable to save skill");
    expect(props.onSuccess).not.toHaveBeenCalled();
    submit();
    await waitFor(() => expect(props.onSuccess).toHaveBeenCalledOnce());
    expect(uploadSkill).toHaveBeenLastCalledWith("sk-test", file, "2.0.0", { skill: undefined, metadata: undefined });
    expect(importSkill).not.toHaveBeenCalled();
    expect(props.onClose).toHaveBeenCalledOnce();
  });

  it.each([
    [new File(["text"], "SKILL.md"), "Please choose a .zip file"],
    [new File([new Uint8Array(10 * 1024 * 1024 + 1)], "large.zip"), "The ZIP file must be 10 MiB or smaller"],
  ])("rejects invalid file case %#", async (invalidFile, message) => {
    renderWithProviders(<AddPluginForm {...props} />);
    chooseFile(invalidFile);
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent(message);
    expect(uploadSkill).not.toHaveBeenCalled();
  });

  it("clears a file on mode change, allows reselection and restores source validation", async () => {
    renderWithProviders(<AddPluginForm {...props} />);
    chooseFile();
    fireEvent.click(screen.getByRole("radio", { name: "Add from URL" }));
    change("Registry name", "review");
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please enter a valid repository or zip archive URL");
    fireEvent.click(screen.getByRole("radio", { name: "Upload ZIP" }));
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please choose a .zip file");
    chooseFile();
    submit();
    await waitFor(() =>
      expect(uploadSkill).toHaveBeenCalledWith("sk-test", file, "", { skill: undefined, metadata: { name: "review" } }),
    );
  });

  it("blocks an invalid version and cancels without publishing", async () => {
    renderWithProviders(<AddPluginForm {...props} />);
    chooseFile();
    change("Version (Optional)", "invalid");
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Version must be in semantic versioning format");
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(props.onClose).toHaveBeenCalledOnce();
    expect(props.onSuccess).not.toHaveBeenCalled();
    expect(uploadSkill).not.toHaveBeenCalled();
  });
});
