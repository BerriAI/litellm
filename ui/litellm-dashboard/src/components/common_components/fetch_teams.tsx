import { teamListCall, Organization } from "../networking";

export const fetchTeams = async (
  accessToken: string,
  userID: string | null,
  userRole: string | null,
  currentOrg: Organization | null,
  setTeams: (teams: any[]) => void,
) => {
  let givenTeams;
  if (userRole != "Admin" && userRole != "Admin Viewer") {
    givenTeams = await teamListCall(accessToken, currentOrg?.organization_id || null, userID);
  } else {
    givenTeams = await teamListCall(accessToken, currentOrg?.organization_id || null);
  }

  console.log(`givenTeams: ${givenTeams}`);

  setTeams(givenTeams);
};
