"use client";

import TransformRequestPanel from "@/components/transform_request";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const APIPlaygroundPage = () => {
  const { accessToken } = useAuthorized();

  return <TransformRequestPanel accessToken={accessToken} />;
};

export default APIPlaygroundPage;
