"use client";

import { AccessGroupsPage } from "./_components/AccessGroupsPage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function AccessGroups() {
  useAuthorized();
  return <AccessGroupsPage />;
}
