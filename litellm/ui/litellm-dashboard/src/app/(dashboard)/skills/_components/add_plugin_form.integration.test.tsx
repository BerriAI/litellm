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
const openImport = () => {
  renderWithProviders(<AddPluginForm {...props} />);
  fireEvent.click(screen.getByRole("radio", { name: "Import URL" }));
};

describe("AddPluginForm", () => {
  beforeEach(() => vi.clearAllMocks());

  it("starts with a portable ZIP and offers host-agnostic imports without guessing metadata", () => {
    renderWithProviders(<AddPluginForm {...props} />);
    expect(screen.getByRole("radio", { name: "Upload ZIP" })).toBeChecked();
    expect(screen.getByText(/frontmatter supplies the skill name and description/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("radio", { name: "Import URL" }));
    expect(screen.getByLabelText("Source URL")).toBeInTheDocument();
    expect(screen.getByLabelText("Path (optional)")).toBeEnabled();
    expect(screen.queryByLabelText("Skill Name")).not.toBeInTheDocument();
  });

  it.each([
    "https://github.com/anthropics/claude-code",
    "https://github.com/anthropics/claude-code/tree/main/plugins/my-skill",
    "https://gitlab.com/group/repo",
    "https://skills-bucket.s3.us-east-1.amazonaws.com/plugins/s3-skill-1.0.0.zip",
  ])("submits the original source URL %s", async (url) => {
    openImport();
    change("Source URL", url);
    submit();
    await waitFor(() => expect(importSkill).toHaveBeenCalledWith("sk-test", { url }));
    expect(props.onSuccess).toHaveBeenCalledOnce();
  });

  it("keeps explicit selector, ref and path choices when a source URL changes", async () => {
    openImport();
    change("Source URL", "https://gitlab.com/group/repo");
    change("Skill selector (optional)", "review");
    change("Git ref (optional)", "release/v2");
    change("Path (optional)", "skills/review");
    change("Version (Optional)", "3.0.0");
    const url = "https://github.com/example/skills/tree/release/v2";
    change("Source URL", url);
    submit();
    const expected = { url, skill: "review", ref: "release/v2", path: "skills/review", version: "3.0.0" };
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
    expect(uploadSkill).toHaveBeenLastCalledWith("sk-test", file, "2.0.0", undefined);
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
    fireEvent.click(screen.getByRole("radio", { name: "Import URL" }));
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please enter a repository or zip archive URL");
    fireEvent.click(screen.getByRole("radio", { name: "Upload ZIP" }));
    submit();
    expect(await screen.findByRole("alert")).toHaveTextContent("Please choose a .zip file");
    chooseFile();
    submit();
    await waitFor(() => expect(uploadSkill).toHaveBeenCalledWith("sk-test", file, "", undefined));
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
