import userEvent from "@testing-library/user-event";
import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { setGlobalLitellmHeaderName } from "@/components/networking";
import type { SkillHubSkill, SkillSubmission } from "@/components/claude_code_plugins/types";
import { toast } from "@/lib/toast";
import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

const published: SkillHubSkill = {
  id: "published-skill",
  name: "shared-skill",
  source: {
    source: "archive",
    url: "http://localhost:4000/skill-hub/skills/shared-skill/archive/approved.zip",
    sha256: "a".repeat(64),
    ref: "approved",
  },
  version: "2.3.4",
  description: "Approved instructions",
  keywords: ["sales, finance", "review"],
  domain: "Engineering",
  namespace: "workflows",
  visibility: "public",
  download_url: "/skill-hub/skills/shared-skill/archive",
  content_url: "/skill-hub/skills/shared-skill/content",
  updated_at: "2026-10-20T00:00:00Z",
  download_count: 0,
};

const pending: SkillSubmission = {
  id: "submission-1",
  revision: 0,
  can_edit: false,
  skill_name: published.name,
  operation: "update",
  status: "pending_review",
  proposal: {
    name: published.name,
    agent_skill: {
      name: published.name,
      description: "Portable instructions",
      digest: "sha256:reviewed",
      origin: "SKILL.md",
    },
    source: { source: "github", repo: "example/skills", ref: "proposed-ref" },
    version: null,
    description: "Proposed instructions",
    keywords: published.keywords,
    domain: published.domain,
    namespace: published.namespace,
  },
  has_archive: true,
  submitted_by: "member@example.org",
  created_at: "2026-10-21T00:00:00Z",
  base_skill_id: published.id,
  base_updated_at: published.updated_at ?? null,
  reviewed_by: null,
  reviewed_at: null,
  rejection_reason: null,
};

const fetchMock = vi.fn<typeof fetch>();
const queryDefaults = testQueryClient.getDefaultOptions();
let catalog: SkillHubSkill[];
let catalogRequests: URL[];
let submissions: SkillSubmission[];
let pageRequests: { limit: number; offset: number }[];
let writes: { path: string; method: string; body: RequestInit["body"] }[];
let reviewFailure: string | null;
let directPublication: boolean;

const respondToWrite = (path: string, body: RequestInit["body"]) => {
  if (reviewFailure && path.startsWith("/skill-hub/submissions/"))
    return Response.json({ detail: { error: reviewFailure } }, { status: 409 });
  if (path.endsWith("/approve") || path.endsWith("/reject")) {
    const approved = path.endsWith("/approve");
    const reason = typeof body === "string" ? JSON.parse(body).reason : null;
    submissions = submissions.map((submission) =>
      submission.id === pending.id
        ? { ...submission, status: approved ? "approved" : "rejected", rejection_reason: reason }
        : submission,
    );
    if (approved) catalog = [{ ...published, ...pending.proposal, version: "2.3.5" }];
    const result = {
      status: "success",
      action: approved ? "approved" : "rejected",
      submission: submissions[0],
      skill: approved ? catalog[0] : null,
    };
    return Response.json(result);
  }
  if (path === `/skill-hub/submissions/${pending.id}`) {
    const metadata = typeof body === "string" ? JSON.parse(body) : {};
    submissions = submissions.map((item) => ({
      ...item,
      revision: (item.revision ?? 0) + 1,
      proposal: { ...item.proposal, ...metadata },
    }));
    return Response.json(submissions[0]);
  }
  if (path.endsWith("/withdraw")) {
    submissions = submissions.map((item) => ({ ...item, status: "withdrawn", can_edit: false }));
    return Response.json(submissions[0]);
  }
  if (directPublication) {
    catalog = [{ ...published, version: "3.0.0", description: "Admin-published instructions" }];
    return Response.json(catalog[0]);
  }
  submissions = [pending];
  const result = {
    status: "pending_review",
    action: "submitted",
    submission_id: pending.id,
    proposal: pending.proposal,
  };
  return Response.json(result);
};

const respondToSubmissionsPage = (url: URL) => {
  const limit = Number(url.searchParams.get("limit") ?? "20");
  const offset = Number(url.searchParams.get("offset") ?? "0");
  pageRequests.push({ limit, offset });
  const page = submissions.slice(offset, offset + limit);
  return Response.json({ submissions: page, count: page.length, has_more: offset + limit < submissions.length });
};

const makeSubmissions = (count: number): SkillSubmission[] =>
  Array.from({ length: count }, (_, index) => ({
    ...pending,
    id: `submission-${index + 1}`,
    skill_name: `skill-${index + 1}`,
    proposal: { ...pending.proposal, description: `Instructions ${index + 1}` },
  }));

let scanFailure = false;
let scanRequested = false;

function scanResponse(method: string) {
  if (method === "POST") scanRequested = true;
  return Response.json({
    configured: true,
    scan: !scanRequested
      ? null
      : {
          id: "scan-1",
          status: scanFailure ? "failed" : "completed",
          requested_by: "admin",
          started_at: "2026-01-01T00:00:00Z",
          expires_at: "2026-01-01T00:03:00Z",
          archive_sha256: "a".repeat(64),
          error: scanFailure ? "Scanner was unreachable" : null,
          report: scanFailure
            ? null
            : {
                scanner_version: "test-version",
                complete: false,
                scan_mode: "static-only",
                risk_score: 85,
                findings: [
                  {
                    rule_id: "unsafe-upload",
                    severity: "HIGH",
                    message: "Potential data exfiltration",
                    file: "SKILL.md",
                    start_line: 8,
                  },
                ],
                limitations: [],
              },
        },
  });
}

beforeEach(() => {
  scanFailure = false;
  scanRequested = false;
  testQueryClient.clear();
  vi.clearAllMocks();
  catalog = [published];
  catalogRequests = [];
  submissions = [pending];
  pageRequests = [];
  writes = [];
  reviewFailure = null;
  directPublication = false;
  fetchMock.mockReset().mockImplementation(async (input, options) => {
    const request = input instanceof Request ? input : null;
    const url = new URL(request?.url ?? String(input), "http://localhost");
    const path = url.pathname;
    const method = request?.method ?? options?.method ?? "GET";
    if (path === `/skill-hub/submissions/${pending.id}/scan`) {
      return scanResponse(method);
    }
    if (method !== "GET") {
      const body = request ? await request.text() : options?.body;
      writes.push({ path, method, body });
      return respondToWrite(path, body);
    }
    if (path === "/skill-hub/skills") {
      catalogRequests.push(url);
      const limit = Number(url.searchParams.get("limit"));
      const offset = Number(url.searchParams.get("offset"));
      const skills = catalog.slice(offset, offset + limit);
      const result = {
        tags: ["review", "engineering"],
        skills,
        count: skills.length,
        next_offset: offset + limit < catalog.length ? offset + limit : null,
      };
      return Response.json(result);
    }
    if (path === `/skill-hub/skills/${published.name}`)
      return Response.json(catalog.find((skill) => skill.name === published.name));
    if (path === `/skill-hub/skills/${published.name}/versions`)
      return Response.json({
        versions: [
          {
            id: "published-version",
            version: published.version,
            description: "Recorded release",
            manifest: { source: published.source },
            published_at: published.updated_at,
            published_by: "maintainer@example.org",
            content_kind: "stored_archive",
            download_url: "/skill-hub/skills/shared-skill/versions/published-version/archive",
          },
        ],
        next_offset: null,
      });
    if (path === "/skill-hub/submissions") return respondToSubmissionsPage(url);
    if (path.endsWith("/archive")) return new Response("zip-bytes", { headers: { "Content-Type": "application/zip" } });
    if (path === `/skill-hub/submissions/${pending.id}`)
      return Response.json({ submission: submissions[0], current_skill: catalog[0] });
    if (path === "/.well-known/litellm-ui-config")
      return Response.json({ claude_code_marketplace_auth_required: true });
    return Response.json({ detail: "Unexpected request" }, { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  testQueryClient.setDefaultOptions(queryDefaults);
  setGlobalLitellmHeaderName();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

const renderSubmissions = (accessToken = "admin-token") =>
  renderWithProviders(<ClaudeCodePluginsPanel accessToken={accessToken} userRole="Admin" />, {
    searchParams: "tab=submissions",
  });

const openEdit = async () => {
  fireEvent.click(await screen.findByText(published.name));
  fireEvent.click(await screen.findByRole("button", { name: "Edit Skill" }));
  return screen.getByRole("dialog");
};

it.each(["current", "retained"])("refreshes published counts after a %s download", async (kind) => {
  URL.createObjectURL = vi.fn(() => "blob:skill-download");
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="admin-token" userRole="Admin" />);
  expect(await screen.findByRole("cell", { name: "0" })).toBeInTheDocument();
  expect(fetchMock).toHaveBeenCalledOnce();
  fireEvent.click(screen.getByText(published.name));
  await screen.findByRole("button", { name: "Edit Skill" });
  if (kind === "retained") fireEvent.click(screen.getByRole("tab", { name: "Version history" }));
  const button = await screen.findByRole("button", {
    name: kind === "retained" ? "Download retained ZIP" : "Download ZIP",
  });
  catalog = [{ ...published, download_count: 1 }];
  fireEvent.click(button);
  await waitFor(() => expect(button).toBeEnabled());
  fireEvent.click(screen.getByText("Skills Hub"));
  expect(await screen.findByRole("cell", { name: "1" })).toBeInTheDocument();
});

it("paginates the neutral catalog and refreshes detail for a recreated skill", async () => {
  catalog = [
    published,
    ...Array.from({ length: 50 }, (_, index) => ({
      ...published,
      id: `skill-${index + 1}`,
      name: `catalog-${index + 1}`,
    })),
  ];
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />);
  fireEvent.click(await screen.findByText(published.name));
  expect(await screen.findByRole("button", { name: "Edit Skill" })).toBeInTheDocument();
  fireEvent.click(screen.getByText("Skills Hub"));
  catalog = [
    { ...published, id: "filler", name: "catalog-0" },
    ...catalog.slice(1),
    { ...published, id: "recreated", description: "Recreated instructions" },
  ];
  expect(await screen.findByText("catalog-49")).toBeInTheDocument();
  expect(screen.queryByText("catalog-50")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("catalog-50")).toBeInTheDocument();
  expect(screen.queryByText("catalog-49")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  fireEvent.click(screen.getByText(published.name));
  expect(await screen.findByRole("button", { name: "Edit Skill" })).toBeInTheDocument();
  expect(screen.getByText("Recreated instructions")).toBeInTheDocument();
  fireEvent.click(screen.getByText("Skills Hub"));
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  await waitFor(() => expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled());
});

it.each(["descriptor", "archive"])("requires a portable %s for a submitted ZIP preview", async (missing) => {
  submissions = [
    {
      ...pending,
      has_archive: missing !== "archive",
      proposal: { ...pending.proposal, agent_skill: missing === "descriptor" ? null : pending.proposal.agent_skill },
    },
  ];
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  expect(await screen.findByText("Currently published")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Download submitted ZIP" })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Approve and publish" })).toBeEnabled();
});

it("paginates more than 20 submissions without showing the prior page while the next request loads or fails", async () => {
  submissions = makeSubmissions(21);
  const defaults = { ...queryDefaults, queries: { ...queryDefaults.queries, staleTime: 0 } };
  testQueryClient.setDefaultOptions(defaults);
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
    searchParams: "tab=submissions",
    onUrlUpdate,
  });
  expect(await screen.findByText("Instructions 20")).toBeInTheDocument();
  expect(screen.queryByText("Instructions 21")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
  const fetchStub = fetchMock.getMockImplementation()!;
  const response = Promise.withResolvers<Response>();
  fetchMock.mockImplementationOnce(async () => response.promise);
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("Page 2")).toBeInTheDocument();
  expect(screen.getByText("Loading submissions...")).toBeInTheDocument();
  expect(screen.queryByText("Instructions 1")).not.toBeInTheDocument();
  expect(screen.queryByText("Instructions 20")).not.toBeInTheDocument();
  await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
  expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("submissions_page")).toBe("2");
  expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("tab")).toBe("submissions");
  await act(async () => response.resolve(Response.json({ detail: "Unavailable" }, { status: 500 })));
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not refresh submissions");
  expect(screen.queryByText("Instructions 20")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  fetchMock.mockImplementation(fetchStub);
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByText("Instructions 21")).toBeInTheDocument();
  expect(screen.queryByText("Instructions 20")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  expect(await screen.findByText("Instructions 1")).toBeInTheDocument();
  expect(screen.queryByText("Instructions 21")).not.toBeInTheDocument();
  await waitFor(() => expect(pageRequests.at(-1)).toEqual({ limit: 20, offset: 0 }));
  expect(pageRequests).toEqual([
    { limit: 20, offset: 0 },
    { limit: 20, offset: 20 },
    { limit: 20, offset: 0 },
  ]);
});

it("keeps differently sized deep-linked pages in separate caches", async () => {
  submissions = makeSubmissions(25);
  const first = renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
    searchParams: "tab=submissions&submissions_page=2",
  });
  expect(await screen.findByText("Instructions 21")).toBeInTheDocument();
  first.unmount();
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
    searchParams: "tab=submissions&submissions_page=2&submissions_page_size=5",
  });
  expect(await screen.findByText("Instructions 6")).toBeInTheDocument();
  expect(screen.queryByText("Instructions 21")).not.toBeInTheDocument();
  expect(pageRequests).toEqual([
    { limit: 20, offset: 20 },
    { limit: 5, offset: 5 },
  ]);
});

it("refreshes inactive submission pages after a review", async () => {
  submissions = makeSubmissions(21);
  renderSubmissions();
  await screen.findByText("Instructions 1");
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("Instructions 21")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Previous" }));
  await screen.findByText("Instructions 1");
  submissions = submissions.map((submission) =>
    submission.id === "submission-21"
      ? { ...submission, status: "rejected", rejection_reason: "Reviewed by another admin" }
      : submission,
  );
  fireEvent.click(screen.getAllByRole("button", { name: "Review" })[0]);
  fireEvent.click(await screen.findByRole("button", { name: "Approve and publish" }));
  await screen.findByText("Approved");
  fireEvent.click(screen.getByRole("button", { name: "Next" }));
  expect(await screen.findByText("Rejection reason: Reviewed by another admin")).toBeInTheDocument();
  expect(screen.getByText("Rejected")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
});

it("allows a member submission, explains approval and refreshes their pending list", async () => {
  submissions = [];
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />);
  await screen.findByText("Approved instructions");
  fireEvent.click(screen.getByRole("tab", { name: "My submissions" }));
  await screen.findByText("No submissions yet");
  fireEvent.click(screen.getByRole("tab", { name: "Published skills" }));
  await waitFor(() =>
    expect(screen.getByRole("tab", { name: "Published skills" })).toHaveAttribute("aria-selected", "true"),
  );
  fireEvent.click(screen.getByRole("button", { name: "+ Register your skill" }));
  fireEvent.click(screen.getByRole("radio", { name: "Import URL" }));
  fireEvent.change(screen.getByRole("textbox", { name: "Source URL" }), {
    target: { value: "https://github.com/example/new-skill" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Submit for approval" }));
  await waitFor(() =>
    expect(toast.success).toHaveBeenCalledWith("Submitted for admin approval. The published skill is unchanged"),
  );
  expect(await screen.findByText("Approved instructions")).toBeInTheDocument();
  expect(screen.queryByText("Proposed instructions")).not.toBeInTheDocument();
  expect(writes[0]).toMatchObject({
    path: "/skill-hub/skills/import",
    method: "POST",
  });
  fireEvent.click(screen.getByRole("tab", { name: "My submissions" }));
  expect(await screen.findByText("Pending review")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "View details" }));
  expect(await screen.findByText("Awaiting admin approval. The published skill remains unchanged")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Approve and publish" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
});

it("retries failed detail loading before enabling edits", async () => {
  const fetchStub = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation((input, options) =>
    String(input).endsWith(`/skill-hub/skills/${published.name}`)
      ? Promise.resolve(Response.json({ detail: "Unavailable" }, { status: 500 }))
      : fetchStub(input, options),
  );
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />);
  fireEvent.click(await screen.findByText(published.name));
  expect(await screen.findByRole("alert")).toHaveTextContent("Retry before editing");
  expect(screen.queryByRole("button", { name: "Edit Skill" })).not.toBeInTheDocument();
  fetchMock.mockImplementation(fetchStub);
  fireEvent.click(screen.getByRole("button", { name: "Retry" }));
  expect(await screen.findByRole("button", { name: "Edit Skill" })).toBeInTheDocument();
});

it("edits the loaded detail with automatic version intent and preserves the approved skill while review is pending", async () => {
  const loaded = { ...published, updated_at: "2026-10-21T12:00:00Z" };
  const fetchStub = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation((input, options) =>
    options?.method === "GET" && String(input).endsWith(`/skill-hub/skills/${published.name}`)
      ? Promise.resolve(Response.json(loaded))
      : fetchStub(input, options),
  );
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />);
  fireEvent.click(await screen.findByText(published.name));
  fireEvent.click(screen.getByText("Version history"));
  expect(await screen.findByText("Recorded release")).toBeInTheDocument();
  fireEvent.click(await screen.findByRole("button", { name: "Edit Skill" }));
  const form = screen.getByRole("dialog");
  expect(within(form).getByLabelText("Version override (optional)")).toHaveValue("");
  expect(within(form).getByLabelText("Namespace (optional)")).toHaveValue(published.namespace);
  fireEvent.change(within(form).getByLabelText("Description (Optional)"), {
    target: { value: "Proposed instructions" },
  });
  fireEvent.click(within(form).getByRole("button", { name: "Submit for approval" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(writes[0]).toMatchObject({ path: "/skill-hub/skills/shared-skill", method: "PUT" });
  const expectedMetadata = {
    version: null,
    expected_updated_at: loaded.updated_at,
    domain: published.domain,
    namespace: published.namespace,
    keywords: published.keywords,
    description: "Proposed instructions",
  };
  expect(JSON.parse(String(writes[0].body))).toMatchObject(expectedMetadata);
  expect(await screen.findByText("Recorded release")).toBeInTheDocument();
  expect(screen.queryByText("Proposed instructions")).not.toBeInTheDocument();
});

it("sends ZIP replacement through PUT upload with automatic version intent and a concurrency timestamp", async () => {
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />);
  const form = await openEdit();
  fireEvent.click(within(form).getByRole("radio", { name: "Upload ZIP" }));
  const file = new File(["new instructions"], "replacement.zip", { type: "application/zip" });
  fireEvent.change(within(form).getByLabelText("Skill ZIP file"), { target: { files: [file] } });
  fireEvent.click(within(form).getByRole("button", { name: "Submit for approval" }));
  await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
  expect(writes[0]).toMatchObject({ path: "/skill-hub/skills/shared-skill/upload", method: "PUT" });
  const body = writes[0].body as FormData;
  expect(body.get("file")).toBe(file);
  expect(body.get("version")).toBeNull();
  expect(body.get("expected_updated_at")).toBe(published.updated_at);
  expect(body.get("metadata")).toBeNull();
  expect(screen.getByText("Approved instructions")).toBeInTheDocument();
});

it.each(["metadata", "upload"])(
  "keeps direct admin %s publication and wires an explicit version override",
  async (sourceType) => {
    directPublication = true;
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="admin-token" userRole="Admin" />);
    const form = await openEdit();
    fireEvent.change(within(form).getByLabelText("Version override (optional)"), { target: { value: "3.0.0" } });
    fireEvent.change(within(form).getByLabelText("Description (Optional)"), {
      target: { value: "Admin-published instructions" },
    });
    if (sourceType === "upload") {
      fireEvent.click(within(form).getByRole("radio", { name: "Upload ZIP" }));
      fireEvent.change(within(form).getByLabelText("Skill ZIP file"), {
        target: { files: [new File(["skill"], "replacement.zip", { type: "application/zip" })] },
      });
    }
    fireEvent.click(within(form).getByRole("button", { name: "Save changes" }));
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(await screen.findByText("Admin-published instructions")).toBeInTheDocument();
    expect(toast.success).toHaveBeenCalledWith("Skill updated successfully");
    expect(screen.queryByText("Submitted for admin approval")).not.toBeInTheDocument();
    const body = writes[0].body;
    if (body instanceof FormData) {
      expect(body.get("version")).toBe("3.0.0");
      expect(body.get("expected_updated_at")).toBe(published.updated_at);
    } else {
      expect(JSON.parse(String(body))).toMatchObject({ version: "3.0.0", expected_updated_at: published.updated_at });
    }
  },
);

it("lets an admin inspect full proposed values, download the private ZIP, approve and see refreshed published details", async () => {
  const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
  vi.stubGlobal(
    "URL",
    Object.assign(URL, { createObjectURL: vi.fn(() => "blob:submission"), revokeObjectURL: vi.fn() }),
  );
  setGlobalLitellmHeaderName("X-LiteLLM-Key");
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  expect(await screen.findByText("Currently published")).toBeInTheDocument();
  const dialog = within(screen.getByRole("dialog"));
  expect(dialog.getByText("Approved instructions")).toBeInTheDocument();
  expect(dialog.getByText("Proposed instructions")).toBeInTheDocument();
  expect(dialog.getByText("Next patch version on approval")).toBeInTheDocument();
  expect(dialog.getByText(/proposed-ref/)).toBeInTheDocument();
  expect(dialog.getByText(/sha256:reviewed/)).toBeInTheDocument();
  expect(dialog.getAllByText(published.namespace!)).toHaveLength(2);
  fireEvent.click(dialog.getByRole("button", { name: "Download submitted ZIP" }));
  await waitFor(() => expect(click).toHaveBeenCalledOnce());
  expect(fetchMock).toHaveBeenCalledWith(
    expect.stringContaining(`/submissions/${pending.id}/archive`),
    expect.objectContaining({
      headers: expect.objectContaining({ "X-LiteLLM-Key": "Bearer admin-token" }),
      redirect: "error",
    }),
  );
  fireEvent.click(dialog.getByRole("button", { name: "Approve and publish" }));
  expect(await screen.findByText("Approved")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole("tab", { name: "Published skills" }));
  await waitFor(() =>
    expect(screen.getByRole("tab", { name: "Published skills" })).toHaveAttribute("aria-selected", "true"),
  );
  expect(await screen.findByRole("cell", { name: "Proposed instructions" })).toBeInTheDocument();
  expect(screen.queryByText("Approved instructions")).not.toBeInTheDocument();
  fireEvent.click(screen.getByText(published.name));
  expect(screen.getByText("Proposed instructions")).toBeInTheDocument();
});

it("saves an admin rejection reason for the author and keeps the published metadata", async () => {
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  fireEvent.change(await screen.findByLabelText("Rejection reason (optional)"), {
    target: { value: "Please include setup instructions" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Reject" }));
  expect(await screen.findByText("Rejection reason: Please include setup instructions")).toBeInTheDocument();
  expect(screen.getByText("Rejected")).toBeInTheDocument();
  expect(writes[0]).toMatchObject({
    path: `/skill-hub/submissions/${pending.id}/reject`,
    method: "PUT",
    body: JSON.stringify({ expected_revision: 0, reason: "Please include setup instructions" }),
  });
  fireEvent.click(screen.getByRole("tab", { name: "Published skills" }));
  await waitFor(() =>
    expect(screen.getByRole("tab", { name: "Published skills" })).toHaveAttribute("aria-selected", "true"),
  );
  expect(await screen.findByRole("cell", { name: "Approved instructions" })).toBeInTheDocument();
});

it("refreshes detail after a review conflict and removes actions when another reviewer has already acted", async () => {
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  await screen.findByRole("button", { name: "Approve and publish" });
  reviewFailure = "Submission has already been reviewed";
  submissions = [{ ...pending, status: "rejected", rejection_reason: "Already reviewed elsewhere" }];
  fireEvent.click(screen.getByRole("button", { name: "Approve and publish" }));
  expect(await screen.findByRole("alert")).toHaveTextContent(reviewFailure);
  await waitFor(() => expect(screen.queryByRole("button", { name: "Approve and publish" })).not.toBeInTheDocument());
  expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  expect(
    within(screen.getByRole("dialog")).getByText("Rejection reason: Already reviewed elsewhere"),
  ).toBeInTheDocument();
});

it("keeps an effective Admin viewer read-only while allowing full submission inspection", async () => {
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="viewer-token" userRole="Admin" isViewOnly />);
  expect(screen.getByRole("button", { name: "+ Register your skill" })).toBeDisabled();
  fireEvent.click(await screen.findByText(published.name));
  expect(screen.queryByRole("button", { name: "Edit Skill" })).not.toBeInTheDocument();
  fireEvent.click(screen.getByText("Skills Hub"));
  fireEvent.click(screen.getByRole("tab", { name: "Submissions" }));
  fireEvent.click(await screen.findByRole("button", { name: "View details" }));
  expect(await screen.findByText("Currently published")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Approve and publish" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  expect(writes).toEqual([]);
});

it.each(["account", "role", "read-only"])(
  "does not show cached submissions after a change of %s when refresh fails",
  async (change) => {
    const { rerender } = renderSubmissions("first-token");
    expect(await screen.findByText("Proposed instructions")).toBeInTheDocument();
    fetchMock.mockImplementation(async () => Response.json({ detail: "Unavailable" }, { status: 500 }));
    await act(async () =>
      rerender(
        <ClaudeCodePluginsPanel
          accessToken={change === "account" ? "second-token" : "first-token"}
          userRole={change === "role" ? "Internal User" : "Admin"}
          isViewOnly={change === "read-only"}
        />,
      ),
    );
    expect(screen.queryByText("Proposed instructions")).not.toBeInTheDocument();
    expect(screen.queryByText("Submitted by member@example.org")).not.toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not refresh submissions");
  },
);

it.each(["role", "read-only"])("does not reuse a published admin snapshot after a change of %s", async (change) => {
  const { rerender } = renderWithProviders(<ClaudeCodePluginsPanel accessToken="same-token" userRole="Admin" />);
  fireEvent.click(await screen.findByText(published.name));
  expect(await screen.findByRole("button", { name: "Edit Skill" })).toBeInTheDocument();
  fetchMock.mockImplementation(async () => Response.json({ detail: "Unavailable" }, { status: 500 }));
  await act(async () =>
    rerender(
      <ClaudeCodePluginsPanel
        accessToken="same-token"
        userRole={change === "role" ? "Internal User" : "Admin"}
        isViewOnly={change === "read-only"}
      />,
    ),
  );
  expect(screen.queryByText("Approved instructions")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Edit Skill" })).not.toBeInTheDocument();
  expect(await screen.findByRole("alert")).toHaveTextContent("Could not refresh published skills");
  expect(screen.getByRole("columnheader", { name: "Downloads" })).toBeInTheDocument();
});

it("does not reuse admin review metadata when the same account loses its role and detail refresh fails", async () => {
  const { rerender } = renderSubmissions("same-token");
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  expect(await screen.findByText("Currently published")).toBeInTheDocument();
  expect(within(screen.getByRole("dialog")).getByText("Approved instructions")).toBeInTheDocument();
  const fetchStub = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (input, options) => {
    const path = new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
    if (path === `/skill-hub/submissions/${pending.id}`)
      return Response.json({ detail: "Unavailable" }, { status: 500 });
    return fetchStub(input, options);
  });
  await act(async () => rerender(<ClaudeCodePluginsPanel accessToken="same-token" userRole="Internal User" />));
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  fireEvent.click(await screen.findByRole("button", { name: "View details" }));
  const dialog = within(screen.getByRole("dialog"));
  expect(dialog.queryByText("Approved instructions")).not.toBeInTheDocument();
  expect(dialog.queryByText("Proposed instructions")).not.toBeInTheDocument();
  expect(dialog.queryByRole("button", { name: "Approve and publish" })).not.toBeInTheDocument();
  expect(await dialog.findByRole("alert")).toHaveTextContent("Could not refresh submissions");
});

it("ignores an in-flight admin submission list that resolves after the same account loses its role", async () => {
  const response = Promise.withResolvers<Response>();
  const requested = Promise.withResolvers<void>();
  const fetchStub = fetchMock.getMockImplementation()!;
  fetchMock.mockImplementation(async (input, options) => {
    const path = new URL(input instanceof Request ? input.url : String(input), "http://localhost").pathname;
    if (path === "/skill-hub/submissions") {
      requested.resolve();
      return response.promise;
    }
    return fetchStub(input, options);
  });
  const { rerender } = renderSubmissions("same-token");
  await requested.promise;
  submissions = [];
  fetchMock.mockImplementation(fetchStub);
  await act(async () => rerender(<ClaudeCodePluginsPanel accessToken="same-token" userRole="Internal User" />));
  expect(await screen.findByText("No submissions yet")).toBeInTheDocument();
  await act(async () => {
    response.resolve(Response.json({ submissions: [pending], count: 1, has_more: false }));
    await response.promise;
  });
  expect(screen.getByText("No submissions yet")).toBeInTheDocument();
  expect(screen.queryByText("Proposed instructions")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Review" })).not.toBeInTheDocument();
});

it.each([false, true])("keeps approval available after manual scanning, failure=%s", async (failure) => {
  scanFailure = failure;
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  const scan = await screen.findByRole("button", { name: "Scan for threats" });
  await waitFor(() => expect(scan).toBeEnabled());
  expect(screen.getByRole("button", { name: "Approve and publish" })).toBeEnabled();
  fireEvent.click(scan);
  expect(
    await screen.findByText(failure ? "Scanner was unreachable" : "Potential data exfiltration"),
  ).toBeInTheDocument();
  if (!failure) {
    expect(screen.getByText("SKILL.md:8")).toBeInTheDocument();
    expect(
      screen.getByText("Analysis was incomplete. This report does not establish that the skill is safe."),
    ).toBeInTheDocument();
  }
  expect(screen.getByRole("button", { name: "Reject" })).toBeEnabled();
  fireEvent.click(screen.getByRole("button", { name: "Approve and publish" }));
  expect(await screen.findByText("Approved")).toBeInTheDocument();
  expect(writes).toHaveLength(1);
  expect(writes[0].path).toBe(`/skill-hub/submissions/${pending.id}/approve`);
});

it.each([200, 409])("refreshes the running scan after POST %s and ignores an older GET", async (status) => {
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  const scan = await screen.findByRole("button", { name: "Scan for threats" });
  await waitFor(() => expect(scan).toBeEnabled());
  const older = Promise.withResolvers<Response>();
  const refreshed = Promise.withResolvers<Response>();
  const running = {
    configured: true,
    scan: {
      id: "current-scan",
      status: "running",
      requested_by: "another-admin",
      started_at: "2026-01-01T00:00:00Z",
      expires_at: "2026-01-01T00:03:00Z",
    },
  };
  const originalFetch = fetchMock.getMockImplementation()!;
  let reads = 0;
  fetchMock.mockImplementation((input, options) => {
    const url = input instanceof Request ? input.url : String(input);
    if (!url.endsWith(`/submissions/${pending.id}/scan`)) return originalFetch(input, options);
    if ((input instanceof Request ? input.method : options?.method) === "POST") {
      return Promise.resolve(
        Response.json(status === 409 ? { detail: "A scan is already running" } : running, { status }),
      );
    }
    return ++reads === 1 ? older.promise : refreshed.promise;
  });
  void testQueryClient.refetchQueries({ type: "active" });
  await waitFor(() => expect(reads).toBe(1));
  fireEvent.click(scan);
  await waitFor(() => expect(reads).toBe(2));
  expect(screen.getByRole("button", { name: "Scanning…" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Approve and publish" })).toBeEnabled();
  refreshed.resolve(Response.json(running));
  expect(await screen.findByText("Requested by another-admin")).toBeInTheDocument();
  await act(async () => older.resolve(Response.json({ configured: true, scan: null })));
  expect(screen.getByRole("button", { name: "Scanning…" })).toBeDisabled();
  expect(screen.getByText("Requested by another-admin")).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "Approve and publish" }));
  expect(await screen.findByText("Approved")).toBeInTheDocument();
});

it("applies tag and scan filters across the catalog and resets pagination when searching", async () => {
  const user = userEvent.setup();
  catalog = [published, { ...published, id: "second", name: "second-skill" }];
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
    searchParams: { skills_page: "2", skills_page_size: "1" },
  });
  await waitFor(() => expect(catalogRequests.at(-1)?.searchParams.get("offset")).toBe("1"));
  await screen.findByRole("cell", { name: "Approved instructions" });
  await user.click(screen.getByRole("button", { name: "Downloads" }));
  await waitFor(() => expect(catalogRequests.at(-1)?.searchParams.get("sort_by")).toBe("download_count"));
  expect(catalogRequests.at(-1)?.searchParams.get("offset")).toBe("0");
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Filter by tag" }), "engineering");
  await waitFor(() => expect(catalogRequests.at(-1)?.searchParams.get("tag")).toBe("engineering"));
  expect(catalogRequests.at(-1)?.searchParams.get("offset")).toBe("0");
  await chooseSelectOption(user, screen.getByRole("combobox", { name: "Security scan" }), "Security scan: Yes");
  await waitFor(() => expect(catalogRequests.at(-1)?.searchParams.get("security_scan_passed")).toBe("true"));
  fireEvent.change(screen.getByRole("textbox", { name: /Search skills/ }), {
    target: { value: "reviewer@example.test" },
  });
  await waitFor(() => expect(catalogRequests.at(-1)?.searchParams.get("query")).toBe("reviewer@example.test"));
  expect(catalogRequests.at(-1)?.searchParams.get("tag")).toBe("engineering");
  expect(catalogRequests.at(-1)?.searchParams.get("security_scan_passed")).toBe("true");
});

it("lets an owner edit the viewed proposal and withdraw it with its current revision", async () => {
  submissions = [{ ...pending, can_edit: true, revision: 3, proposal: { ...pending.proposal, version: "3.0.0" } }];
  renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
    searchParams: "tab=submissions",
  });
  fireEvent.click(await screen.findByRole("button", { name: "View details" }));
  fireEvent.click(await screen.findByRole("button", { name: "Edit submission" }));
  expect(screen.getByLabelText("Version override (optional)")).toHaveValue("3.0.0");
  expect(
    screen.getByText(
      `Current version ${published.version}. Leave blank for the next patch version, or enter a higher version`,
    ),
  ).toBeInTheDocument();
  fireEvent.change(await screen.findByLabelText("Description (Optional)"), {
    target: { value: "Owner revised instructions" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Save submission" }));
  expect(await screen.findByRole("cell", { name: "Owner revised instructions" })).toBeInTheDocument();
  expect(JSON.parse(String(writes[0].body))).toMatchObject({
    expected_revision: 3,
    description: "Owner revised instructions",
  });
  expect(catalog).toEqual([published]);
  fireEvent.click(screen.getByRole("button", { name: "Withdraw submission" }));
  expect(writes).toHaveLength(1);
  fireEvent.click(await screen.findByRole("button", { name: "Confirm withdrawal" }));
  expect(await screen.findByText("Withdrawn")).toBeInTheDocument();
  expect(writes[1]).toMatchObject({
    path: `/skill-hub/submissions/${pending.id}/withdraw`,
    body: JSON.stringify({ expected_revision: 4 }),
  });
  fireEvent.click(screen.getByRole("button", { name: "View details" }));
  expect(screen.queryByRole("button", { name: "Edit submission" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Withdraw submission" })).not.toBeInTheDocument();
});

it("keeps admin review separate from ownership and sends the displayed revision", async () => {
  submissions = [{ ...pending, revision: 7, can_edit: false }];
  renderSubmissions();
  fireEvent.click(await screen.findByRole("button", { name: "Review" }));
  expect(screen.queryByRole("button", { name: "Edit submission" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Withdraw submission" })).not.toBeInTheDocument();
  fireEvent.click(await screen.findByRole("button", { name: "Approve and publish" }));
  expect(await screen.findByText("Approved")).toBeInTheDocument();
  expect(JSON.parse(String(writes[0].body))).toEqual({ expected_revision: 7 });
});

it.each(["metadata", "upload"])(
  "reloads a conflicted %s edit only when the author discards the draft",
  async (mode) => {
    submissions = [{ ...pending, can_edit: true }];
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="member-token" userRole="Internal User" />, {
      searchParams: "tab=submissions",
    });
    fireEvent.click(await screen.findByRole("button", { name: "View details" }));
    fireEvent.click(await screen.findByRole("button", { name: "Edit submission" }));
    fireEvent.change(screen.getByLabelText("Description (Optional)"), { target: { value: "Unsaved draft" } });
    if (mode === "upload") {
      fireEvent.click(screen.getByRole("radio", { name: "Upload ZIP" }));
      fireEvent.change(screen.getByLabelText("Skill ZIP file"), {
        target: { files: [new File(["replacement"], "skill.zip", { type: "application/zip" })] },
      });
    }
    submissions = [
      { ...pending, can_edit: true, revision: 1, proposal: { ...pending.proposal, description: "Latest edit" } },
    ];
    reviewFailure = "This submission changed. Reload it before trying again.";
    fireEvent.click(screen.getByRole("button", { name: "Save submission" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(reviewFailure);
    expect(screen.getByRole("dialog", { name: "Edit submission" })).toBeInTheDocument();
    reviewFailure = null;
    fireEvent.click(screen.getByRole("button", { name: "Discard draft and reload submission" }));
    expect(await screen.findByRole("cell", { name: "Latest edit" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Edit submission" }));
    expect(screen.getByLabelText("Description (Optional)")).toHaveValue("Latest edit");
    fireEvent.click(screen.getByRole("button", { name: "Save submission" }));
    await waitFor(() => expect(writes).toHaveLength(2));
    expect(JSON.parse(String(writes[1].body))).toMatchObject({ expected_revision: 1, description: "Latest edit" });
  },
);
