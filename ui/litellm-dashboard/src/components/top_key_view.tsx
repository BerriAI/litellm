import React, { useState } from "react";
import { BarChart } from "@tremor/react";
import KeyInfoView from "./key_info_view";
import { keyInfoV1Call } from "./networking";
import { transformKeyInfo } from "../components/key_team_helpers/transform_key_info";

interface TopKeyViewProps {
  topKeys: any[];
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  teams: any[] | null;
}

const TopKeyView: React.FC<TopKeyViewProps> = ({ 
  topKeys, 
  accessToken, 
  userID, 
  userRole,
  teams
}) => {
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [keyData, setKeyData] = useState<any | undefined>(undefined);

  const handleKeyClick = async (item: any) => {
    if (!accessToken) return;
    
    try {
      const keyInfo = await keyInfoV1Call(accessToken, item.api_key);
      const transformedKeyData = transformKeyInfo(keyInfo);
      setKeyData(transformedKeyData);
      setSelectedKey(item.key);
      setIsModalOpen(true);  // Open modal when key is clicked
    } catch (error) {
      console.error("Error fetching key info:", error);
    }
  };

  const handleClose = () => {
    setIsModalOpen(false);
    setSelectedKey(null);
    setKeyData(undefined);
  };

  return (
    <>
      <BarChart
        className="mt-4 h-40 cursor-pointer"
        data={topKeys}
        index="key"
        categories={["spend"]}
        colors={["cyan"]}
        yAxisWidth={80}
        tickGap={5}
        layout="vertical"
        showXAxis={false}
        showLegend={false}
        valueFormatter={(value) => `$${value.toFixed(2)}`}
        onValueChange={(item) => handleKeyClick(item)}
      />

      {isModalOpen && selectedKey && keyData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-11/12 max-w-6xl max-h-[90vh] overflow-y-auto">
            <KeyInfoView
              keyId={selectedKey}
              onClose={handleClose}
              keyData={keyData}
              accessToken={accessToken}
              userID={userID}
              userRole={userRole}
              teams={teams}
            />
          </div>
        </div>
      )}
    </>
  );
};

export default TopKeyView;
