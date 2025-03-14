import React, { useState } from "react";
import { BarChart } from "@tremor/react";
import KeyInfoView from "./key_info_view";
import { keyInfoCall } from "./networking";

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
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [keyData, setKeyData] = useState<any | undefined>(undefined);

  const handleKeyClick = async (item: any) => {
    if (!accessToken || !item.key) return;
    
    try {
      // Get the full key data using the API key
      const keyInfo = await keyInfoCall(accessToken, [item.key]);
      if (keyInfo && keyInfo.keys && keyInfo.keys.length > 0) {
        setKeyData(keyInfo.keys[0]);
        setSelectedKey(item.key);
      }
    } catch (error) {
      console.error("Error fetching key info:", error);
    }
  };

  const handleClose = () => {
    setSelectedKey(null);
    setKeyData(undefined);
  };

  if (selectedKey && keyData) {
    return (
      <KeyInfoView
        keyId={selectedKey}
        onClose={handleClose}
        keyData={keyData}
        accessToken={accessToken}
        userID={userID}
        userRole={userRole}
        teams={teams}
      />
    );
  }

  return (
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
  );
};

export default TopKeyView;
