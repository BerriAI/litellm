"use client";

import { AccessGroupsPage } from "@/components/AccessGroups/AccessGroupsPage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function AccessGroups() {
  useAuthorized();
  return <AccessGroupsPage />;
}
