import { screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import PolicySelector, { getPolicyOptionEntries, policyVersionRef, POLICY_VERSION_ID_PREFIX } from "./PolicySelector";
import { Policy } from "./types";

vi.mock("../networking");

const makePolicy = (overrides: Partial<Policy>): Policy => ({
  policy_id: "uuid-1",
  policy_name: "test-policy",
  inherit: null,
  description: null,
  guardrails_add: [],
  guardrails_remove: [],
  condition: null,
  ...overrides,
});

describe("policyVersionRef", () => {
  it("should prefix the policy id with the version prefix", () => {
    expect(policyVersionRef("abc-123")).toBe(`${POLICY_VERSION_ID_PREFIX}abc-123`);
  });
});

describe("getPolicyOptionEntries", () => {
  it("should filter out draft policies", () => {
    const policies = [
      makePolicy({ policy_name: "draft-one", version_status: "draft" }),
      makePolicy({ policy_name: "published-one", version_status: "published", policy_id: "pub-id" }),
    ];
    const options = getPolicyOptionEntries(policies);
    expect(options).toHaveLength(1);
    expect(options[0].label).toContain("published-one");
  });

  it("should use the policy_name as value for production policies", () => {
    const policy = makePolicy({ policy_name: "prod-policy", version_status: "production" });
    const options = getPolicyOptionEntries([policy]);
    expect(options[0].value).toBe("prod-policy");
  });

  it("should use a version ref as value for published (non-production) policies", () => {
    const policy = makePolicy({ policy_id: "abc-123", policy_name: "pub-policy", version_status: "published" });
    const options = getPolicyOptionEntries([policy]);
    expect(options[0].value).toBe(policyVersionRef("abc-123"));
  });

  it("should include the version number and status in the label", () => {
    const policy = makePolicy({ policy_name: "my-policy", version_status: "published", version_number: 3 });
    const options = getPolicyOptionEntries([policy]);
    expect(options[0].label).toContain("v3");
    expect(options[0].label).toContain("published");
  });

  it("should append the description to the label when present", () => {
    const policy = makePolicy({
      policy_name: "my-policy",
      version_status: "published",
      description: "blocks PII",
    });
    const options = getPolicyOptionEntries([policy]);
    expect(options[0].label).toContain("blocks PII");
  });

  it("should treat policies with no version_status as draft and filter them out", () => {
    const policy = makePolicy({ policy_name: "implicit-draft" });
    const options = getPolicyOptionEntries([policy]);
    expect(options).toHaveLength(0);
  });
});

describe("PolicySelector", () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    renderWithProviders(
      <PolicySelector accessToken="tok" onChange={mockOnChange} />
    );
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should fetch policies on mount with the given access token", async () => {
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    renderWithProviders(<PolicySelector accessToken="my-token" onChange={mockOnChange} />);
    await waitFor(() => {
      expect(networking.getPoliciesList).toHaveBeenCalledWith("my-token");
    });
  });

  it("should call onPoliciesLoaded with the fetched policies after mount", async () => {
    const policies = [makePolicy({ version_status: "production" })];
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies });
    const onPoliciesLoaded = vi.fn();
    renderWithProviders(
      <PolicySelector accessToken="tok" onChange={mockOnChange} onPoliciesLoaded={onPoliciesLoaded} />
    );
    await waitFor(() => {
      expect(onPoliciesLoaded).toHaveBeenCalledWith(policies);
    });
  });

  it("should show a disabled placeholder when disabled prop is true", () => {
    vi.mocked(networking.getPoliciesList).mockResolvedValue({ policies: [] });
    renderWithProviders(
      <PolicySelector accessToken="tok" onChange={mockOnChange} disabled />
    );
    expect(screen.getByRole("combobox")).toBeDisabled();
  });

  it("should not fetch policies when accessToken is empty", () => {
    renderWithProviders(<PolicySelector accessToken="" onChange={mockOnChange} />);
    expect(networking.getPoliciesList).not.toHaveBeenCalled();
  });
});
