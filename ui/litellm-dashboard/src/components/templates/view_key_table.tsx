"use client";
import { Setter } from "@/types";
import React from "react";
import { VirtualKeysTable } from "../VirtualKeysPage/VirtualKeysTable";
import useKeyList, { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";

// Define the props type
interface ViewKeyTableProps {
  userID: string | null;
  userRole: string | null;
  accessToken: string | null;
  selectedTeam: Team | null;
  setSelectedTeam: React.Dispatch<React.SetStateAction<any | null>>;
  data: KeyResponse[] | null;
  setData: (keys: KeyResponse[]) => void;
  teams: Team[] | null;
  premiumUser: boolean;
  currentOrg: Organization | null;
  organizations: Organization[] | null;
  setCurrentOrg: React.Dispatch<React.SetStateAction<Organization | null>>;
  selectedKeyAlias: string | null;
  setSelectedKeyAlias: Setter<string | null>;
  createClicked: boolean;
  setAccessToken?: (token: string) => void;
}

const ViewKeyTable: React.FC<ViewKeyTableProps> = ({
  accessToken,
  selectedTeam,
  setSelectedTeam,
  teams,
  currentOrg,
  organizations,
  setCurrentOrg,
  selectedKeyAlias,
  setSelectedKeyAlias,
  createClicked,
}) => {
  const { keys, isLoading, error, pagination, refresh, setKeys } = useKeyList({
    selectedTeam: selectedTeam || undefined,
    currentOrg,
    selectedKeyAlias,
    accessToken: accessToken || "",
    createClicked,
    expand: ["user"],
  });

  const handlePageChange = (newPage: number) => {
    refresh({ page: newPage });
  };

  return (
    <div>
      <VirtualKeysTable teams={teams} organizations={organizations} />
    </div>
  );
};

// Update the type declaration to include the new function
declare global {
  interface Window {
    refreshKeysList?: () => void;
    addNewKeyToList?: (newKey: any) => void;
  }
}

export default ViewKeyTable;
