import { describe, it, expect, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { ProjectKeysSection } from "./ProjectKeysSection";

const mockUseKeys = vi.fn();
vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({
  useKeys: (...args: unknown[]) => mockUseKeys(...args),
}));

vi.mock("@/components/common_components/DefaultProxyAdminTag", () => ({
  default: ({ userId }: { userId: string }) => <span>{userId}</span>,
}));

const emptyKeysResponse = {
  data: { keys: [], total_count: 0, current_page: 1, total_pages: 1 },
  isLoading: false,
};

describe("ProjectKeysSection", () => {
  it("should render", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByRole("table")).toBeInTheDocument();
  });

  it("should show the Keys card title", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByText("Keys")).toBeInTheDocument();
  });

  it("should display the total key count from the API response", () => {
    mockUseKeys.mockReturnValue({
      data: { keys: [], total_count: 42, current_page: 1, total_pages: 9 },
      isLoading: false,
    });
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByText("42 keys")).toBeInTheDocument();
  });

  it("should show 'No keys found' when the project has no keys", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByText("No keys found")).toBeInTheDocument();
  });

  it("should render a search input for filtering by key name", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(screen.getByPlaceholderText("Filter by key name...")).toBeInTheDocument();
  });

  it("should call useKeys with the projectId", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-abc" />);
    expect(mockUseKeys).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      expect.objectContaining({ projectID: "proj-abc" })
    );
  });

  it("should pass null for selectedKeyAlias when the filter input is empty", () => {
    mockUseKeys.mockReturnValue(emptyKeysResponse);
    renderWithProviders(<ProjectKeysSection projectId="proj-1" />);
    expect(mockUseKeys).toHaveBeenCalledWith(
      expect.any(Number),
      expect.any(Number),
      expect.objectContaining({ selectedKeyAlias: null })
    );
  });
});
