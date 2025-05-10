import { Locator, Page } from "@playwright/test";

export class TeamsPage {
  private readonly createNewTeamButton: Locator;
  private readonly teamNameField: Locator;
  private readonly createTeamButton: Locator;

  constructor(private readonly page: Page) {
    this.createNewTeamButton = this.page.getByRole("button", {
      name: "+ Create New Team",
    });
    this.teamNameField = this.page.getByLabel("Team Name");
    this.createTeamButton = this.page.getByRole("button", {
      name: "Create Team",
    });
  }

  getCreateNewTeamButton(): Locator {
    return this.createNewTeamButton;
  }

  getTeamNameField(): Locator {
    return this.teamNameField;
  }

  getCreateTeamButton(): Locator {
    return this.createTeamButton;
  }
}
