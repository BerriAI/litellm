"use client";

import React, { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { ResetPasswordForm } from "./ResetPasswordForm";

function ResetPasswordContent() {
  const searchParams = useSearchParams()!;
  const token = searchParams.get("token");
  return <ResetPasswordForm token={token} />;
}

export default function ResetPassword() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center min-h-screen">Loading...</div>}>
      <ResetPasswordContent />
    </Suspense>
  );
}
