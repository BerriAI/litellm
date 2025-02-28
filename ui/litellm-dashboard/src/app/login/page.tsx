// ... existing imports ...
import { setCookie } from "@/utils/cookies";

export default function Login() {
  // ... existing code ...

  const handleLogin = async (values: any) => {
    // ... existing logic ...
    if (response.token && response.user_id) {
      setCookie("token", response.token, response.user_id);
    }
    // ... rest of the handler
  };

  // ... rest of the component
}