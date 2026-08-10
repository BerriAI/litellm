import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { credentialListCall, vectorStoreListCall } from "@/components/networking";

import VectorStoreManagement from "./index";

vi.mock("@/components/networking", () => ({
  vectorStoreListCall: vi.fn(),
  vectorStoreDeleteCall: vi.fn(),
  credentialListCall: vi.fn(),
}));

vi.mock("./VectorStoreTable", () => ({
  __esModule: true,
  default: ({ isLoading }: { isLoading?: boolean }) => (
    <div data-testid="vector-store-table">{isLoading ? "table-loading" : "table-loaded"}</div>
  ),
}));

vi.mock("./VectorStoreForm", () => ({ __esModule: true, default: () => null }));
vi.mock("./vector_store_info", () => ({ __esModule: true, default: () => null }));
vi.mock("./CreateVectorStore", () => ({ __esModule: true, default: () => null }));
vi.mock("./TestVectorStoreTab", () => ({ __esModule: true, default: () => null }));

const mockVectorStoreListCall = vi.mocked(vectorStoreListCall);
const mockCredentialListCall = vi.mocked(credentialListCall);

const openManageTab = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(screen.getByRole("tab", { name: "Manage Vector Stores" }));
};

describe("VectorStoreManagement loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    const user = userEvent.setup();
    render(<VectorStoreManagement accessToken={null} userID={null} userRole={null} />);
    await openManageTab(user);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockVectorStoreListCall).not.toHaveBeenCalled();
  });

  it("should show the loading state until the vector store fetch settles", async () => {
    const user = userEvent.setup();
    let resolveFetch: (value: { data: never[] }) => void = () => {};
    mockVectorStoreListCall.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole="Admin" />);
    await openManageTab(user);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ data: [] });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test");
  });
});

describe("VectorStoreManagement credential access", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({ data: [] });
    mockCredentialListCall.mockResolvedValue({ credentials: [] });
  });

  it.each(["Internal User", "Internal Viewer", "Org Admin"])(
    "should not call GET /credentials for %s",
    async (userRole) => {
      const user = userEvent.setup();
      render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole={userRole} />);
      await openManageTab(user);

      expect(await screen.findByText("table-loaded")).toBeInTheDocument();
      expect(mockVectorStoreListCall).toHaveBeenCalledWith("sk-test");
      expect(mockCredentialListCall).not.toHaveBeenCalled();
    },
  );

  it.each(["Admin", "Admin Viewer"])("should call GET /credentials for %s", async (userRole) => {
    const user = userEvent.setup();
    render(<VectorStoreManagement accessToken="sk-test" userID="user-1" userRole={userRole} />);
    await openManageTab(user);

    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockCredentialListCall).toHaveBeenCalledWith("sk-test");
  });
});
