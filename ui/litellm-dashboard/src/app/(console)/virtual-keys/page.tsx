"use client";

import VirtualKeysTable from "@/app/(console)/virtual-keys/components/VirtualKeysTable/VirtualKeysTable";
import { getCookie } from "@/utils/cookieUtils";
import { jwtDecode } from "jwt-decode";
import { useState } from "react";
import useKeyList from "@/components/key_team_helpers/key_list";
import { useRouter } from "next/navigation";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Col, Grid } from "@tremor/react";
import CreateKey from "@/app/(console)/virtual-keys/components/CreateKey";
import useAuthorized from "@/app/(console)/hooks/useAuthorized"

const VirtualKeysPage = () => {
  const {accessToken, userRole} = useAuthorized();
  const [createClicked, setCreateClicked] = useState<boolean>(false);

  const queryClient = new QueryClient();

  const { keys, isLoading, error, pagination, refresh, setKeys } = useKeyList({
    selectedKeyAlias: null,
    currentOrg: null,
    accessToken: accessToken || "",
    createClicked,
  });

  const addKey = (data: any) => {
    setKeys((prevData) => (prevData ? [...prevData, data] : [data]));
    setCreateClicked(() => !createClicked);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <div className="w-full mx-4 h-[75vh]">
        <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
          <Col numColSpan={1} className="flex flex-col gap-2">
            <CreateKey team={null} userRole={userRole} data={null} addKey={addKey} />
            <VirtualKeysTable
              keys={keys}
              setKeys={setKeys}
              pagination={pagination}
              onPageChange={() => {}}
              organizations={null}
            />
          </Col>
        </Grid>
      </div>
    </QueryClientProvider>
  );
};

export default VirtualKeysPage;
