import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi, beforeEach } from "vitest";
import UserSearchModal from "./user_search_modal";
import { userFilterUICall } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  userFilterUICall: vi.fn(),
}));

const mockedFilterCall = userFilterUICall as unknown as ReturnType<typeof vi.fn>;

const openModal = (
  overrides?: Partial<React.ComponentProps<typeof UserSearchModal>>,
) =>
  render(
    <UserSearchModal
      isVisible={true}
      onCancel={vi.fn()}
      onSubmit={vi.fn(async () => {})}
      accessToken="sk-test"
      {...overrides}
    />,
  );

// Antd Select renders a hidden <input role="combobox">. There are three:
// email, user-id, role. We grab the first two by their dataset hooks.
const getEmailInput = () =>
  document.querySelector(
    '[data-testid="member-email-search"] input.ant-select-selection-search-input',
  )! as HTMLInputElement;
const getUserIdInput = () =>
  document.querySelector(
    '[data-testid="member-userid-search"] input.ant-select-selection-search-input',
  )! as HTMLInputElement;

// Antd renders the open dropdown as a separate floating div with role="listbox".
// Use this to scope queries so we don't pick up the value rendered inside the
// closed select's display chip (which also has the option text).
const getOpenDropdown = (): HTMLElement => {
  const dropdowns = document.querySelectorAll(".ant-select-dropdown");
  for (const el of Array.from(dropdowns)) {
    if (!el.classList.contains("ant-select-dropdown-hidden")) {
      return el as HTMLElement;
    }
  }
  // Fall back to the first one if we can't tell which is open
  return dropdowns[0] as HTMLElement;
};

describe("UserSearchModal — manual entry fallback (LIT-3389)", () => {
  beforeEach(() => {
    mockedFilterCall.mockReset();
  });

  it("offers a 'Use email' fallback option when /user/filter/ui errors so team admins can still add a member by email", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn(async () => {});
    mockedFilterCall.mockRejectedValue(
      Object.assign(new Error("Forbidden"), { status: 403 }),
    );

    openModal({ onSubmit });

    await user.click(getEmailInput());
    await user.type(getEmailInput(), "alice@example.com");

    await waitFor(
      () =>
        expect(
          screen.getByText('Use email "alice@example.com"'),
        ).toBeInTheDocument(),
      { timeout: 2000 },
    );

    await user.click(screen.getByText('Use email "alice@example.com"'));
    await user.click(screen.getByRole("button", { name: /Add Member/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      user_email: "alice@example.com",
      user_id: "",
      role: "user",
    });
  });

  it("offers a 'Use user ID' fallback option when the search returns no matches", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn(async () => {});
    mockedFilterCall.mockResolvedValue([]);

    openModal({ onSubmit });

    await user.click(getUserIdInput());
    await user.type(getUserIdInput(), "user-1234");

    await waitFor(
      () =>
        expect(screen.getByText('Use user ID "user-1234"')).toBeInTheDocument(),
      { timeout: 2000 },
    );

    await user.click(screen.getByText('Use user ID "user-1234"'));
    await user.click(screen.getByRole("button", { name: /Add Member/i }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toMatchObject({
      user_id: "user-1234",
      user_email: "",
      role: "user",
    });
  });

  it("does not duplicate the synthetic option when the API returns an exact match", async () => {
    const user = userEvent.setup();
    mockedFilterCall.mockResolvedValue([
      { user_id: "u-1", user_email: "bob@example.com" },
    ]);

    openModal();

    await user.click(getEmailInput());
    await user.type(getEmailInput(), "bob@example.com");

    // Wait until the dropdown has at least one option matching the API result.
    await waitFor(
      () => {
        const dropdown = getOpenDropdown();
        expect(
          within(dropdown).getAllByText("bob@example.com").length,
        ).toBeGreaterThan(0);
      },
      { timeout: 2000 },
    );

    // Only one option in the dropdown — no synthetic duplicate.
    const dropdown = getOpenDropdown();
    expect(
      within(dropdown).queryByText('Use email "bob@example.com"'),
    ).not.toBeInTheDocument();
  });

  it("appends the synthetic option alongside partial API matches", async () => {
    const user = userEvent.setup();
    mockedFilterCall.mockResolvedValue([
      { user_id: "u-1", user_email: "bobby@example.com" },
    ]);

    openModal();

    await user.click(getEmailInput());
    await user.type(getEmailInput(), "bob");

    await waitFor(
      () => {
        const dropdown = getOpenDropdown();
        expect(
          within(dropdown).getAllByText("bobby@example.com").length,
        ).toBeGreaterThan(0);
        expect(
          within(dropdown).getByText('Use email "bob"'),
        ).toBeInTheDocument();
      },
      { timeout: 2000 },
    );
  });

  it("clears the dropdown when the search box is cleared", async () => {
    const user = userEvent.setup();
    mockedFilterCall.mockResolvedValue([]);

    openModal();

    await user.click(getEmailInput());
    await user.type(getEmailInput(), "x");
    await waitFor(
      () => expect(screen.getByText('Use email "x"')).toBeInTheDocument(),
      { timeout: 2000 },
    );

    await user.keyboard("{Backspace}");
    await waitFor(
      () =>
        expect(
          screen.queryByText('Use email "x"'),
        ).not.toBeInTheDocument(),
      { timeout: 2000 },
    );
  });

  it("ignores whitespace-only input (does not surface a 'Use \"   \"' option)", async () => {
    const user = userEvent.setup();
    mockedFilterCall.mockResolvedValue([]);

    openModal();

    await user.click(getEmailInput());
    await user.type(getEmailInput(), "   ");

    await new Promise((r) => setTimeout(r, 600));
    expect(screen.queryByText(/Use email "\s*"/i)).not.toBeInTheDocument();
  });

  it("renders nothing-to-select dropdown when isVisible=false (smoke)", () => {
    openModal({ isVisible: false });
    expect(
      screen.queryByRole("button", { name: /Add Member/i }),
    ).not.toBeInTheDocument();
  });
});
