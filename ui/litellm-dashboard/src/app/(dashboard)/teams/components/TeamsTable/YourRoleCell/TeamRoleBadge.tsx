import { ShieldIcon, UserIcon } from "lucide-react";

const MEMBER_BADGE_BG = "#F3F4F6"; // gray-100
const MEMBER_BADGE_TEXT = "#4B5563"; // gray-600
const MEMBER_BADGE_BORDER = "#E5E7EB"; // gray-200

const ADMIN_BADGE_BG = "#EEF2FF"; // indigo-50
const ADMIN_BADGE_TEXT = "#3730A3"; // indigo-800
const ADMIN_BADGE_BORDER = "#C7D2FE"; // indigo-200

const TeamRoleBadge = (role: string | null) => {
  const base = "inline-flex items-center px-2.5 py-0.5 rounded-md text-xs font-medium border";

  switch (role) {
    case "admin":
      return (
        <span
          className={base}
          style={{
            backgroundColor: ADMIN_BADGE_BG,
            color: ADMIN_BADGE_TEXT,
            borderColor: ADMIN_BADGE_BORDER,
          }}
        >
          <ShieldIcon className="h-3 w-3 mr-1" />
          Admin
        </span>
      );
    case "user":
    default:
      return (
        <span
          className={base}
          style={{
            backgroundColor: MEMBER_BADGE_BG,
            color: MEMBER_BADGE_TEXT,
            borderColor: MEMBER_BADGE_BORDER,
          }}
        >
          <UserIcon className="h-3 w-3 mr-1" />
          Member
        </span>
      );
  }
};

export default TeamRoleBadge;
