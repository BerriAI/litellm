import { Router } from "next/router";
import { getCookie } from "@/utils/cookieUtils";
import { useRouter } from "next/navigation";
import { jwtDecode } from "jwt-decode";

const useAuthorized = () => {
  const token = getCookie("token");
  const router = useRouter();

  if (!token) {
    router.replace("/sso/key/generate");
  }

  const decoded = jwtDecode(token as string) as { [key: string]: any };
  const accessToken = decoded.key;
  const userId = decoded.user_id;
  const userRole = decoded.user_role;
  const premiumUser = decoded.premium_user;

  return { accessToken, userId, userRole, premiumUser };
};

export default useAuthorized;
