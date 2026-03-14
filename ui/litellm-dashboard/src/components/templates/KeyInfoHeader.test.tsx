import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { KeyInfoHeader, KeyInfoData } from "./KeyInfoHeader";

const MOCK_DATA: KeyInfoData = {
  keyName: "My Test Key",
  keyId: "sk-1234567890abcdef",
  userId: "user-abc-123",
  userEmail: "test@example.com",
  createdBy: "admin@example.com",
  createdAt: "Oct 29, 2025 at 1:26 AM",
  lastUpdated: "Oct 29, 2025 at 1:47 AM",
  lastActive: "Oct 29, 2025 at 2:00 AM",
};

describe("KeyInfoHeader", () => {
  it("should render", () => {
    render(<KeyInfoHeader data={MOCK_DATA} />);
    expect(screen.getByText("My Test Key")).toBeInTheDocument();
  });

  it("should render the key ID with prefix", () => {
    render(<KeyInfoHeader data={MOCK_DATA} />);
    expect(screen.getByText(/Key ID:/)).toBeInTheDocument();
    expect(screen.getByText(/sk-1234567890abcdef/)).toBeInTheDocument();
  });

  it("should render all metadata fields", () => {
    render(<KeyInfoHeader data={MOCK_DATA} />);
    expect(screen.getByText("User Email")).toBeInTheDocument();
    expect(screen.getByText("test@example.com")).toBeInTheDocument();
    expect(screen.getByText("User ID")).toBeInTheDocument();
    expect(screen.getByText("user-abc-123")).toBeInTheDocument();
    expect(screen.getByText("Created At")).toBeInTheDocument();
    expect(screen.getByText("Created By")).toBeInTheDocument();
    expect(screen.getByText("Last Updated")).toBeInTheDocument();
    expect(screen.getByText("Last Active")).toBeInTheDocument();
  });

  describe("back button", () => {
    it("should render with default text", () => {
      render(<KeyInfoHeader data={MOCK_DATA} />);
      expect(screen.getByRole("button", { name: /back to keys/i })).toBeInTheDocument();
    });

    it("should render with custom text", () => {
      render(<KeyInfoHeader data={MOCK_DATA} backButtonText="Back to Dashboard" />);
      expect(screen.getByRole("button", { name: /back to dashboard/i })).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /back to keys/i })).not.toBeInTheDocument();
    });

    it("should call onBack when clicked", async () => {
      const onBack = vi.fn();
      render(<KeyInfoHeader data={MOCK_DATA} onBack={onBack} />);
      await userEvent.click(screen.getByRole("button", { name: /back to keys/i }));
      expect(onBack).toHaveBeenCalledTimes(1);
    });
  });

  describe("action buttons", () => {
    it("should show Regenerate and Delete buttons by default", () => {
      render(<KeyInfoHeader data={MOCK_DATA} />);
      expect(screen.getByRole("button", { name: /regenerate key/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /delete key/i })).toBeInTheDocument();
    });

    it("should show Regenerate and Delete buttons when canModifyKey is true", () => {
      render(<KeyInfoHeader data={MOCK_DATA} canModifyKey={true} />);
      expect(screen.getByRole("button", { name: /regenerate key/i })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /delete key/i })).toBeInTheDocument();
    });

    it("should hide Regenerate and Delete buttons when canModifyKey is false", () => {
      render(<KeyInfoHeader data={MOCK_DATA} canModifyKey={false} />);
      expect(screen.queryByRole("button", { name: /regenerate key/i })).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /delete key/i })).not.toBeInTheDocument();
    });

    it("should call onRegenerate when Regenerate Key is clicked", async () => {
      const onRegenerate = vi.fn();
      render(<KeyInfoHeader data={MOCK_DATA} onRegenerate={onRegenerate} />);
      await userEvent.click(screen.getByRole("button", { name: /regenerate key/i }));
      expect(onRegenerate).toHaveBeenCalledTimes(1);
    });

    it("should call onDelete when Delete Key is clicked", async () => {
      const onDelete = vi.fn();
      render(<KeyInfoHeader data={MOCK_DATA} onDelete={onDelete} />);
      await userEvent.click(screen.getByRole("button", { name: /delete key/i }));
      expect(onDelete).toHaveBeenCalledTimes(1);
    });

    it("should disable Regenerate button when regenerateDisabled is true", () => {
      render(<KeyInfoHeader data={MOCK_DATA} regenerateDisabled={true} />);
      expect(screen.getByRole("button", { name: /regenerate key/i })).toBeDisabled();
    });

    it("should not disable Regenerate button by default", () => {
      render(<KeyInfoHeader data={MOCK_DATA} />);
      expect(screen.getByRole("button", { name: /regenerate key/i })).not.toBeDisabled();
    });
  });

  describe("Create New Key button", () => {
    it("should show when onCreateNew is provided", () => {
      render(<KeyInfoHeader data={MOCK_DATA} onCreateNew={vi.fn()} />);
      expect(screen.getByRole("button", { name: /create new key/i })).toBeInTheDocument();
    });

    it("should hide when onCreateNew is not provided", () => {
      render(<KeyInfoHeader data={MOCK_DATA} />);
      expect(screen.queryByRole("button", { name: /create new key/i })).not.toBeInTheDocument();
    });

    it("should call onCreateNew when clicked", async () => {
      const onCreateNew = vi.fn();
      render(<KeyInfoHeader data={MOCK_DATA} onCreateNew={onCreateNew} />);
      await userEvent.click(screen.getByRole("button", { name: /create new key/i }));
      expect(onCreateNew).toHaveBeenCalledTimes(1);
    });
  });

  describe("default_user_id handling", () => {
    it("should show Default Proxy Admin tag for User ID when value is default_user_id", () => {
      const data = { ...MOCK_DATA, userId: "default_user_id" };
      render(<KeyInfoHeader data={data} />);
      expect(screen.getAllByText("Default Proxy Admin").length).toBeGreaterThanOrEqual(1);
    });

    it("should show Default Proxy Admin tag for Created By when value is default_user_id", () => {
      const data = { ...MOCK_DATA, createdBy: "default_user_id" };
      render(<KeyInfoHeader data={data} />);
      expect(screen.getAllByText("Default Proxy Admin").length).toBeGreaterThanOrEqual(1);
    });
  });

  describe("empty value handling", () => {
    it("should show '-' for User Email when value is empty", () => {
      const data = { ...MOCK_DATA, userEmail: "" };
      render(<KeyInfoHeader data={data} />);
      expect(screen.getByText("-")).toBeInTheDocument();
    });
  });
});
