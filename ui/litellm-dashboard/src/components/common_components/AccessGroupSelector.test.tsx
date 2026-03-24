import { renderWithProviders, screen } from "../../../tests/test-utils";
import { vi, describe, it, expect, beforeEach } from "vitest";
import AccessGroupSelector from "./AccessGroupSelector";

const mockAccessGroups = [
  {
    access_group_id: "ag-1",
    access_group_name: "Engineering",
    description: null,
    access_model_names: [],
    access_mcp_server_ids: [],
    access_agent_ids: [],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2024-01-01",
    created_by: null,
    updated_at: "2024-01-01",
    updated_by: null,
  },
  {
    access_group_id: "ag-2",
    access_group_name: "Product",
    description: null,
    access_model_names: [],
    access_mcp_server_ids: [],
    access_agent_ids: [],
    assigned_team_ids: [],
    assigned_key_ids: [],
    created_at: "2024-01-01",
    created_by: null,
    updated_at: "2024-01-01",
    updated_by: null,
  },
];

const mockUseAccessGroups = vi.fn();

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => mockUseAccessGroups(),
  // Re-export the type for imports to work
  get AccessGroupResponse() {
    return {};
  },
}));

describe("AccessGroupSelector", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("should render", () => {
    mockUseAccessGroups.mockReturnValue({
      data: mockAccessGroups,
      isLoading: false,
      isError: false,
    });

    renderWithProviders(<AccessGroupSelector />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should show a loading skeleton when data is loading", () => {
    mockUseAccessGroups.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
    });

    renderWithProviders(<AccessGroupSelector />);
    // Ant Design Skeleton renders with class ant-skeleton
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
  });

  it("should show the label text when showLabel is true", () => {
    mockUseAccessGroups.mockReturnValue({
      data: mockAccessGroups,
      isLoading: false,
      isError: false,
    });

    renderWithProviders(
      <AccessGroupSelector showLabel labelText="My Groups" />
    );
    expect(screen.getByText("My Groups")).toBeInTheDocument();
  });

  it("should not show the label by default", () => {
    mockUseAccessGroups.mockReturnValue({
      data: mockAccessGroups,
      isLoading: false,
      isError: false,
    });

    renderWithProviders(<AccessGroupSelector />);
    expect(screen.queryByText("Access Group")).not.toBeInTheDocument();
  });

  it("should show error content when data fails to load", async () => {
    mockUseAccessGroups.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });

    const user = (await import("@testing-library/user-event")).default.setup();
    renderWithProviders(<AccessGroupSelector />);

    // Open the dropdown to see notFoundContent
    await user.click(screen.getByRole("combobox"));
    expect(
      await screen.findByText("Failed to load access groups")
    ).toBeInTheDocument();
  });
});
