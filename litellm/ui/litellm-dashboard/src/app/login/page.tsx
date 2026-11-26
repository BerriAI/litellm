"use client";

import LoginPage from "./LoginPage";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useControlPlane } from "@/contexts/ControlPlaneContext";
import { uiHref } from "@/utils/uiHref";
import LoadingScreen from "@/components/common_components/LoadingScreen";

export default function Page() {
  const controlPlane = useControlPlane();
  const router = useRouter();
  useEffect(() => {
    if (controlPlane) router.replace(uiHref("api-keys/"));
  }, [controlPlane, router]);
  return controlPlane ? <LoadingScreen /> : <LoginPage />;
}
