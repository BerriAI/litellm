import { AlertTriangleIcon, XIcon } from "lucide-react";
import React, { useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";

interface DeleteTeamModalProps {
  teams: Team[] | null;
  teamToDelete: string | null;
  onCancel: () => void;
  onConfirm: () => void;
}

const DeleteTeamModal = ({ teams, teamToDelete, onCancel, onConfirm }: DeleteTeamModalProps) => {
  const [deleteConfirmInput, setDeleteConfirmInput] = useState("");

  const team = teams?.find((t) => t.team_id === teamToDelete);
  const teamName = team?.team_alias || "";
  const keyCount = team?.keys?.length || 0;
  const isValid = deleteConfirmInput === teamName;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl min-h-[380px] py-6 overflow-hidden transform transition-all flex flex-col justify-between">
        <div>
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-semibold text-gray-900">Delete Team</h3>
            <button
              onClick={() => {
                onCancel();
                setDeleteConfirmInput("");
              }}
              className="text-gray-400 hover:text-gray-500 focus:outline-none"
            >
              <XIcon size={20} />
            </button>
          </div>
          <div className="px-6 py-4">
            {keyCount > 0 && (
              <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-100 rounded-md mb-5">
                <div className="text-red-500 mt-0.5">
                  <AlertTriangleIcon size={20} />
                </div>
                <div>
                  <p className="text-base font-medium text-red-600">
                    Warning: This team has {keyCount} associated key{keyCount > 1 ? "s" : ""}.
                  </p>
                  <p className="text-base text-red-600 mt-2">
                    Deleting the team will also delete all associated keys. This action is irreversible.
                  </p>
                </div>
              </div>
            )}
            <p className="text-base text-gray-600 mb-5">
              Are you sure you want to force delete this team and all its keys?
            </p>
            <div className="mb-5">
              <label className="block text-base font-medium text-gray-700 mb-2">
                {`Type `}
                <span className="underline">{teamName}</span>
                {` to confirm deletion:`}
              </label>
              <input
                type="text"
                value={deleteConfirmInput}
                onChange={(e) => setDeleteConfirmInput(e.target.value)}
                placeholder="Enter team name exactly"
                className="w-full px-4 py-3 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-base"
                autoFocus
              />
            </div>
          </div>
        </div>
        <div className="px-6 py-4 bg-gray-50 flex justify-end gap-4">
          <button
            onClick={() => {
              onCancel();
              setDeleteConfirmInput("");
            }}
            className="px-5 py-3 bg-white border border-gray-300 rounded-md text-base font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={!isValid}
            className={`px-5 py-3 rounded-md text-base font-medium text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 ${isValid ? "bg-red-600 hover:bg-red-700" : "bg-red-300 cursor-not-allowed"}`}
          >
            Force Delete
          </button>
        </div>
      </div>
    </div>
  );
};

export default DeleteTeamModal;
