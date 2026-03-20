import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import OrganizationDropdown from "./OrganizationDropdown";

const MOCK_ORGS = [
  {
    organization_id: "org-1",
    organization_alias: "Engineering",
    budget_id: "",
    metadata: {},
    models: [],
    spend: 0,
    model_spend: {},
    created_at: "",
    created_by: "",
    updated_at: "",
  },
  {
    organization_id: "org-2",
    organization_alias: "Sales",
    budget_id: "",
    metadata: {},
    models: [],
    spend: 0,
    model_spend: {},
    created_at: "",
    created_by: "",
    updated_at: "",
  },
];

describe("OrganizationDropdown", () => {
  it("should render", () => {
    render(<OrganizationDropdown organizations={MOCK_ORGS} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should display organization options when opened", async () => {
    const user = userEvent.setup();
    render(<OrganizationDropdown organizations={MOCK_ORGS} />);

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByText("Engineering")).toBeInTheDocument();
    expect(screen.getByText("Sales")).toBeInTheDocument();
  });

  it("should call onChange with the org id when an organization is selected", async () => {
    const onChange = vi.fn();
    const user = userEvent.setup();
    render(<OrganizationDropdown organizations={MOCK_ORGS} onChange={onChange} />);

    await user.click(screen.getByRole("combobox"));
    await user.click(await screen.findByText("Engineering"));

    expect(onChange).toHaveBeenCalledWith("org-1", expect.anything());
  });

  it("should add ant-select-disabled class when disabled prop is true", () => {
    const { container } = render(<OrganizationDropdown organizations={MOCK_ORGS} disabled={true} />);
    expect(container.querySelector(".ant-select-disabled")).toBeTruthy();
  });

  it("should render with empty organizations list", () => {
    render(<OrganizationDropdown organizations={[]} />);
    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });
});
