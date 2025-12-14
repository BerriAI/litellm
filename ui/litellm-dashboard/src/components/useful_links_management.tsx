import React, { useState, useEffect } from "react";
import { Modal } from "antd";
import { PlusCircleIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { isAdminRole } from "../utils/roles";
import { getPublicModelHubInfo, updateUsefulLinksCall, getProxyBaseUrl } from "./networking";
import { Card, Title, Text, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";
import NotificationsManager from "./molecules/notifications_manager";
import TableIconActionButton from "./common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";

interface UsefulLinksManagementProps {
  accessToken: string | null;
  userRole: string | null;
}

interface Link {
  id: string;
  displayName: string;
  url: string;
  index?: number;
}

const UsefulLinksManagement: React.FC<UsefulLinksManagementProps> = ({ accessToken, userRole }) => {
  const [links, setLinks] = useState<Link[]>([]);
  const [newLink, setNewLink] = useState({ url: "", displayName: "" });
  const [editingLink, setEditingLink] = useState<Link | null>(null);
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isRearranging, setIsRearranging] = useState(false);
  const [originalLinksOrder, setOriginalLinksOrder] = useState<Link[]>([]);

  const fetchUsefulLinks = async () => {
    if (!accessToken) return;

    try {
      setLoading(true);
      const response = await getPublicModelHubInfo();

      if (response && response.useful_links) {
        const usefulLinks = response.useful_links || {};

        // Convert object to array of links with ids
        // Handle both old format (Dict[str, str]) and new format (Dict[str, {url, index}])
        const linksArray = Object.entries(usefulLinks)
          .map(([displayName, value]) => {
            // Check if it's the new format with {url, index}
            if (typeof value === "object" && value !== null && "url" in value) {
              return {
                id: `${(value as any).index ?? 0}-${displayName}`,
                displayName,
                url: (value as any).url as string,
                index: (value as any).index ?? 0,
              };
            } else {
              // Old format: just a string URL
              return {
                id: `0-${displayName}`,
                displayName,
                url: value as string,
                index: 0,
              };
            }
          })
          .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
          .map((link, index) => ({
            ...link,
            id: `${index}-${link.displayName}`,
          }));

        setLinks(linksArray);
      } else {
        setLinks([]);
      }
    } catch (error) {
      console.error("Error fetching useful links:", error);
      setLinks([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchUsefulLinks();
  }, [accessToken]);

  // Check if user is admin
  if (!isAdminRole(userRole || "")) {
    return null;
  }

  const saveLinksToBackend = async (updatedLinks: Link[]) => {
    if (!accessToken) return false;

    try {
      // Convert array back to object format with index for ordering
      // New format: { "displayName": { "url": "...", "index": 0 } }
      const linksObject: Record<string, { url: string; index: number }> = {};
      updatedLinks.forEach((link, index) => {
        linksObject[link.displayName] = {
          url: link.url,
          index: index,
        };
      });

      await updateUsefulLinksCall(accessToken, linksObject);
      // show success modal with public model hub link
      Modal.success({
        title: "Links Saved Successfully",
        content: (
          <div className="py-4">
            <p className="text-gray-600 mb-4">
              Your useful links have been saved and are now visible on the public model hub.
            </p>
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <p className="text-sm text-blue-800 mb-2 font-medium">View your updated model hub:</p>
              <a
                href={`${getProxyBaseUrl()}/ui/model_hub_table`}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center text-blue-600 hover:text-blue-800 underline text-sm font-medium"
              >
                Open Public Model Hub →
              </a>
            </div>
          </div>
        ),
        width: 500,
        okText: "Close",
        maskClosable: true,
        keyboard: true,
      });

      return true;
    } catch (error) {
      console.error("Error saving links:", error);
      NotificationsManager.fromBackend(`Failed to save links - ${error}`);
      return false;
    }
  };

  const handleAddLink = async () => {
    if (!newLink.url || !newLink.displayName) return;

    // Validate URL
    try {
      new URL(newLink.url);
    } catch {
      NotificationsManager.fromBackend("Please enter a valid URL");
      return;
    }

    // Check for duplicate display names
    if (links.some((link) => link.displayName === newLink.displayName)) {
      NotificationsManager.fromBackend("A link with this display name already exists");
      return;
    }

    const newLinkObj: Link = {
      id: `${Date.now()}-${newLink.displayName}`,
      displayName: newLink.displayName,
      url: newLink.url,
    };

    const updatedLinks = [...links, newLinkObj];

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      setNewLink({ url: "", displayName: "" });
      NotificationsManager.success("Link added successfully");
    }
  };

  const handleEditLink = (link: Link) => {
    setEditingLink({ ...link });
  };

  const handleUpdateLink = async () => {
    if (!editingLink) return;

    // Validate URL
    try {
      new URL(editingLink.url);
    } catch {
      NotificationsManager.fromBackend("Please enter a valid URL");
      return;
    }

    // Check for duplicate display names (excluding current link)
    if (links.some((link) => link.id !== editingLink.id && link.displayName === editingLink.displayName)) {
      NotificationsManager.fromBackend("A link with this display name already exists");
      return;
    }

    const updatedLinks = links.map((link) => (link.id === editingLink.id ? editingLink : link));

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      setEditingLink(null);
      NotificationsManager.success("Link updated successfully");
    }
  };

  const handleCancelEdit = () => {
    setEditingLink(null);
  };

  const deleteLink = async (linkId: string) => {
    const updatedLinks = links.filter((link) => link.id !== linkId);

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      NotificationsManager.success("Link deleted successfully");
    }
  };

  const setCurrentLink = (url: string) => {
    window.open(url, "_blank");
  };

  const handleStartRearranging = () => {
    if (editingLink) {
      setEditingLink(null);
    }
    setOriginalLinksOrder([...links]);
    setIsRearranging(true);
  };

  const handleCancelRearranging = () => {
    setLinks([...originalLinksOrder]);
    setIsRearranging(false);
    setOriginalLinksOrder([]);
  };

  const handleSaveRearranging = async () => {
    if (await saveLinksToBackend(links)) {
      setIsRearranging(false);
      setOriginalLinksOrder([]);
      NotificationsManager.success("Link order saved successfully");
    }
  };

  const handleMoveUp = (index: number) => {
    if (index === 0) return;
    const newLinks = [...links];
    [newLinks[index - 1], newLinks[index]] = [newLinks[index], newLinks[index - 1]];
    setLinks(newLinks);
  };

  const handleMoveDown = (index: number) => {
    if (index === links.length - 1) return;
    const newLinks = [...links];
    [newLinks[index], newLinks[index + 1]] = [newLinks[index + 1], newLinks[index]];
    setLinks(newLinks);
  };

  return (
    <Card className="mb-6">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex flex-col">
          <Title className="mb-0">Link Management</Title>
          <p className="text-sm text-gray-500">
            Manage the links that are displayed under &apos;Useful Links&apos; on the public model hub.
          </p>
        </div>
        <div className="flex items-center">
          {isExpanded ? (
            <ChevronDownIcon className="w-5 h-5 text-gray-500" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-gray-500" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <div className="mb-6">
            <Text className="text-sm font-medium text-gray-700 mb-2">Add New Link</Text>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-gray-500 mb-1">Display Name</label>
                <input
                  type="text"
                  value={newLink.displayName}
                  onChange={(e) =>
                    setNewLink({
                      ...newLink,
                      displayName: e.target.value,
                    })
                  }
                  placeholder="Friendly name"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1">URL</label>
                <input
                  type="text"
                  value={newLink.url}
                  onChange={(e) =>
                    setNewLink({
                      ...newLink,
                      url: e.target.value,
                    })
                  }
                  placeholder="https://example.com"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddLink}
                  disabled={!newLink.url || !newLink.displayName}
                  className={`flex items-center px-4 py-2 rounded-md text-sm ${!newLink.url || !newLink.displayName ? "bg-gray-300 text-gray-500 cursor-not-allowed" : "bg-green-600 text-white hover:bg-green-700"}`}
                >
                  <PlusCircleIcon className="w-4 h-4 mr-1" />
                  Add Link
                </button>
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between mb-2">
            <Text className="text-sm font-medium text-gray-700">Manage Existing Links</Text>
            {!isRearranging ? (
              <button
                onClick={handleStartRearranging}
                className="text-xs bg-purple-50 text-purple-600 px-3 py-1.5 rounded hover:bg-purple-100 flex items-center"
              >
                Rearrange Order
              </button>
            ) : (
              <div className="flex space-x-2">
                <button
                  onClick={handleSaveRearranging}
                  className="text-xs bg-green-600 text-white px-3 py-1.5 rounded hover:bg-green-700"
                >
                  Save Order
                </button>
                <button
                  onClick={handleCancelRearranging}
                  className="text-xs bg-gray-50 text-gray-600 px-3 py-1.5 rounded hover:bg-gray-100"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell className="py-1 h-8">Display Name</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">URL</TableHeaderCell>
                    <TableHeaderCell className="py-1 h-8">Actions</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {links.map((link, index) => (
                    <TableRow key={link.id} className="h-8">
                      {editingLink && editingLink.id === link.id ? (
                        <>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingLink.displayName}
                              onChange={(e) =>
                                setEditingLink({
                                  ...editingLink,
                                  displayName: e.target.value,
                                })
                              }
                              className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingLink.url}
                              onChange={(e) =>
                                setEditingLink({
                                  ...editingLink,
                                  url: e.target.value,
                                })
                              }
                              className="w-full px-2 py-1 border border-gray-300 rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleUpdateLink}
                                className="text-xs bg-blue-50 text-blue-600 px-2 py-1 rounded hover:bg-blue-100"
                              >
                                Save
                              </button>
                              <button
                                onClick={handleCancelEdit}
                                className="text-xs bg-gray-50 text-gray-600 px-2 py-1 rounded hover:bg-gray-100"
                              >
                                Cancel
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm text-gray-900">{link.displayName}</TableCell>
                          <TableCell className="py-0.5 text-sm text-gray-500">{link.url}</TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            {isRearranging ? (
                              <div className="flex space-x-2">
                                <TableIconActionButton
                                  variant="Up"
                                  onClick={() => handleMoveUp(index)}
                                  tooltipText="Move up"
                                  disabled={index === 0}
                                  disabledTooltipText="Already at the top"
                                  dataTestId={`move-up-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Down"
                                  onClick={() => handleMoveDown(index)}
                                  tooltipText="Move down"
                                  disabled={index === links.length - 1}
                                  disabledTooltipText="Already at the bottom"
                                  dataTestId={`move-down-${link.id}`}
                                />
                              </div>
                            ) : (
                              <div className="flex space-x-2">
                                <TableIconActionButton
                                  variant="Open"
                                  onClick={() => setCurrentLink(link.url)}
                                  tooltipText="Open link"
                                  dataTestId={`open-link-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Edit"
                                  onClick={() => handleEditLink(link)}
                                  tooltipText="Edit link"
                                  dataTestId={`edit-link-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Delete"
                                  onClick={() => deleteLink(link.id)}
                                  tooltipText="Delete link"
                                  dataTestId={`delete-link-${link.id}`}
                                />
                              </div>
                            )}
                          </TableCell>
                        </>
                      )}
                    </TableRow>
                  ))}
                  {links.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={3} className="py-0.5 text-sm text-gray-500 text-center">
                        No links added yet. Add a new link above.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
};

export default UsefulLinksManagement;
