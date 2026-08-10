import { render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { CredentialItem } from "@/components/networking";

const mockUseAuthorized = vi.fn();
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => mockUseAuthorized(),
}));

const storedCredentials: CredentialItem[] = [
  {
    credential_name: "openai-key",
    credential_values: {},
    credential_info: { custom_llm_provider: "openai" },
  },
];

const mockUseCredentials = vi.fn();
vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => mockUseCredentials(),
}));

vi.mock("@/app/(dashboard)/hooks/models/useModelCostMap", () => ({
  useModelCostMap: () => ({ data: {}, isLoading: false, error: null }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({ data: [], isLoading: false, error: null }),
}));

vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return { ...actual, useQueryClient: () => ({ invalidateQueries: vi.fn() }) };
});

const addModelFormProps = vi.fn();
vi.mock("@/components/add_model/AddModelForm", () => ({
  __esModule: true,
  default: (props: { credentials: CredentialItem[] | null }) => {
    addModelFormProps(props);
    return <div data-testid="add-model-form" />;
  },
}));

import AddModelPanel from "./AddModelPanel";

const credentialsPassedToForm = (): CredentialItem[] | null =>
  addModelFormProps.mock.calls.at(-1)?.[0].credentials ?? null;

describe("AddModelPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseCredentials.mockReturnValue({ data: { credentials: storedCredentials } });
  });

  const renderAs = (userRole: string) => {
    mockUseAuthorized.mockReturnValue({ accessToken: "sk-test", userRole, userId: "user-1" });
    render(<AddModelPanel />);
  };

  it.each(["Internal User", "Internal Viewer", "Org Admin"])(
    "hides the credential picker from %s by passing null",
    (userRole) => {
      renderAs(userRole);

      expect(credentialsPassedToForm()).toBeNull();
    },
  );

  it.each(["Admin", "Admin Viewer"])("passes the fetched credentials through for %s", (userRole) => {
    renderAs(userRole);

    expect(credentialsPassedToForm()).toEqual(storedCredentials);
  });
});
