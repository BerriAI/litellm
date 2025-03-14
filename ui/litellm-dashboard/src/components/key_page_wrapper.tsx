import React, { useState } from "react";
import TopKeyView from "./top_key_view";
import KeyInfoView from "./key_info_view";
import { KeyResponse } from "./key_team_helpers/key_list";

interface KeyPageWrapperProps {
  topKeys: any[];
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  teams: any[] | null;
}

const KeyPageWrapper: React.FC<KeyPageWrapperProps> = ({
  topKeys,
  accessToken,
  userID,
  userRole,
  teams,
}) => {
  const [selectedKeyData, setSelectedKeyData] = useState<KeyResponse | null>(null);

  if (selectedKeyData) {
    return (
      <div className="h-full w-full">
        <KeyInfoView
          keyId={selectedKeyData.token}
          keyData={selectedKeyData}
          onClose={() => setSelectedKeyData(null)}
          accessToken={accessToken}
          userID={userID}
          userRole={userRole}
          teams={teams}
        />
      </div>
    );
  }

  return (
    <TopKeyView
      topKeys={topKeys}
      accessToken={accessToken}
      userID={userID}
      userRole={userRole}
      onKeySelect={setSelectedKeyData}
    />
  );
};

export default KeyPageWrapper; 