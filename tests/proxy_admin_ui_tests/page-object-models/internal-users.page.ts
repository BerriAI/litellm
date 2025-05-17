import { Locator, Page } from "@playwright/test";

export class InternalUsersPage {
  private readonly inviteUserButton: Locator;
  private readonly inviteUserFormUserEmailField: Locator;
  private readonly inviteUserFormTeamIdField: Locator;
  private readonly inviteUserFormTeamRoleField: Locator;
  private readonly inviteUserFormTeamRoleUserSelection: Locator;
  private readonly inviteUserFormCreateUserButton: Locator;

  constructor(private readonly page: Page) {
    this.inviteUserButton = this.page.getByRole("button", {
      name: "+ Invite User",
    });
    this.inviteUserFormUserEmailField = this.page.getByLabel("User Email");
    this.inviteUserFormTeamIdField = this.page.getByLabel("Team ID");
    this.inviteUserFormTeamRoleField = this.page.getByLabel("Team Role");
    this.inviteUserFormTeamRoleUserSelection = this.page.getByText(
      "user- User role. Can view"
    );
    this.inviteUserFormCreateUserButton = this.page.getByRole("button", {
      name: "Create User",
    });
  }

  getInviteUsersButton(): Locator {
    return this.inviteUserButton;
  }

  getInviteUserFormUserEmailField(): Locator {
    return this.inviteUserFormUserEmailField;
  }

  getInviteUserFormTeamIdField(): Locator {
    return this.inviteUserFormTeamIdField;
  }

  getInviteUserFormTeamRoleField(): Locator {
    return this.inviteUserFormTeamRoleField;
  }

  getInviteUserFormTeamRoleSelection(selection: string): Locator {
    return this.page.getByText(selection);
  }

  getInviteUserFormTeamRoleUserSelection(): Locator {
    return this.inviteUserFormTeamRoleUserSelection;
  }

  getInviteUserFormCreateUserButton(): Locator {
    return this.inviteUserFormCreateUserButton;
  }
}
