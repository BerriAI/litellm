"use client";

import { useMemo, type ComponentProps } from "react";
import { configureLensHttp, LensHostProvider, LensWorkspace, type LensHost } from "@litellm/lens-ui";
import { LogDetailsDrawer } from "@/components/logs/detail";
import { getGlobalLitellmHeaderName, getProxyBaseUrl, handleError, uiSpendLogsCall } from "@/components/networking";
import { getAuthToken } from "@/lib/http/runtime";
import { serverRootPath } from "@/lib/serverRootPath";
import { useGatewayLensNavigation } from "./GatewayLensNavigation";

const host: LensHost = {
  surface: "embedded",
  analysis: "deployment",
  spendLogs: { lookup: uiSpendLogsCall, Drawer: LogDetailsDrawer },
};

const http = {
  getBaseUrl: getProxyBaseUrl,
  getAuthToken,
  getAuthHeaderName: getGlobalLitellmHeaderName,
  onError: handleError,
  getServerRootPath: () => serverRootPath,
};

configureLensHttp(http);

export function EmbeddedLens(props: ComponentProps<typeof LensWorkspace>) {
  const navigation = useGatewayLensNavigation();
  const embeddedHost = useMemo(() => ({ ...host, navigation }), [navigation]);
  return (
    <LensHostProvider host={embeddedHost}>
      <LensWorkspace {...props} />
    </LensHostProvider>
  );
}
