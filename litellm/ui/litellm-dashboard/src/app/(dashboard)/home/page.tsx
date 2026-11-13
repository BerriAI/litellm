"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import HomePage from "./_components/HomePage";

export default function HomeRoutePage() {
  const { isLoading, isAuthorized } = useAuthorized();
  if (isLoading || !isAuthorized) {
    return <LoadingScreen />;
  }
  return <HomePage />;
}
