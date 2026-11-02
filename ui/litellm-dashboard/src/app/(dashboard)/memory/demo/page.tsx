"use client";

import useCan from "@/app/(dashboard)/hooks/useCan";
import { AdminOnlyNotice } from "@/components/shared/AdminOnlyNotice";
import SharedKnowledgeDemo from "./SharedKnowledgeDemo";

export default function SharedKnowledgeDemoPage() {
  const canViewMemory = useCan("viewMemory");

  if (!canViewMemory) {
    return <AdminOnlyNotice pageTitle="Memory" />;
  }

  return <SharedKnowledgeDemo />;
}
