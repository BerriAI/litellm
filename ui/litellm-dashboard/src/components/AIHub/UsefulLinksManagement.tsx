import TableIconActionButton from "@/components/common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { isAdminRole } from "@/utils/roles";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink as ExternalLinkIcon,
  Plus,
} from "lucide-react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
} from "@tremor/react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import Link from "next/link";
import React, { useCallback, useEffect, useState } from "react";
import {
  getProxyBaseUrl,
  getPublicModelHubInfo,
  updateUsefulLinksCall,
} from "../networking";

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
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [loading, setLoading] = useState(false);
  const [isExpanded, setIsExpanded] = useState(true);
  const [isRearranging, setIsRearranging] = useState(false);
  const [originalLinksOrder, setOriginalLinksOrder] = useState<Link[]>([]);

  const fetchUsefulLinks = useCallback(async () => {
    if (!accessToken) return;

    try {
      setLoading(true);
      const response = await getPublicModelHubInfo();

      if (response && response.useful_links) {
        const usefulLinks = response.useful_links || {};

        const linksArray = Object.entries(usefulLinks)
          .map(([displayName, value]) => {
            if (
              typeof value === "object" &&
              value !== null &&
              "url" in value
            ) {
              const v = value as { url: string; index?: number };
              return {
                id: `${v.index ?? 0}-${displayName}`,
                displayName,
                url: v.url,
                index: v.index ?? 0,
              };
            } else {
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
  }, [accessToken]);

  useEffect(() => {
    fetchUsefulLinks();
  }, [fetchUsefulLinks]);

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

  const addDisabled = !newLink.url || !newLink.displayName;

  return (
    <Card className="mb-6 p-4">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex flex-col">
          <h3 className="mb-0 text-lg font-semibold">Link Management</h3>
          <p className="text-sm text-muted-foreground">
            Manage the links that are displayed under &apos;Useful Links&apos;
            on the public model hub.
          </p>
        </div>
        <div className="flex items-center">
          {isExpanded ? (
            <ChevronDown className="w-5 h-5 text-muted-foreground" />
          ) : (
            <ChevronRight className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <div className="mb-6">
            <p className="text-sm font-medium text-foreground mb-2">
              Add New Link
            </p>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">
                  Display Name
                </label>
                <Input
                  type="text"
                  value={newLink.displayName}
                  onChange={(e) =>
                    setNewLink({
                      ...newLink,
                      displayName: e.target.value,
                    })
                  }
                  placeholder="Friendly name"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">
                  URL
                </label>
                <Input
                  type="text"
                  value={newLink.url}
                  onChange={(e) =>
                    setNewLink({
                      ...newLink,
                      url: e.target.value,
                    })
                  }
                  placeholder="https://example.com"
                />
              </div>
              <div className="flex items-end">
                <Button
                  onClick={handleAddLink}
                  disabled={addDisabled}
                  className={cn(
                    !addDisabled &&
                      "bg-emerald-600 text-white hover:bg-emerald-700",
                  )}
                >
                  <Plus className="h-4 w-4" />
                  Add Link
                </Button>
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-foreground">
              Manage Existing Links
            </p>
            <div className="flex items-center space-x-2">
              <Link
                href={`${getProxyBaseUrl()}/ui/model_hub_table`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs bg-blue-50 text-blue-600 dark:bg-blue-950/30 dark:text-blue-300 px-3 py-1.5 rounded hover:bg-blue-100 dark:hover:bg-blue-950/60 flex items-center"
                title="Open Public Model Hub"
              >
                Public Model Hub
                <ExternalLinkIcon className="w-4 h-4 ml-1" />
              </Link>
              {!isRearranging ? (
                <button
                  type="button"
                  onClick={handleStartRearranging}
                  className="text-xs bg-purple-50 text-purple-600 dark:bg-purple-950/30 dark:text-purple-300 px-3 py-1.5 rounded hover:bg-purple-100 dark:hover:bg-purple-950/60 flex items-center"
                >
                  Rearrange Order
                </button>
              ) : (
                <div className="flex space-x-2">
                  <button
                    type="button"
                    onClick={handleSaveRearranging}
                    className="text-xs bg-emerald-600 text-white px-3 py-1.5 rounded hover:bg-emerald-700"
                  >
                    Save Order
                  </button>
                  <button
                    type="button"
                    onClick={handleCancelRearranging}
                    className="text-xs bg-muted text-muted-foreground px-3 py-1.5 rounded hover:bg-muted/70"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
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
                            <Input
                              type="text"
                              value={editingLink.displayName}
                              onChange={(e) =>
                                setEditingLink({
                                  ...editingLink,
                                  displayName: e.target.value,
                                })
                              }
                              className="h-8"
                            />
                          </TableCell>
                          <TableCell className="py-0.5">
                            <Input
                              type="text"
                              value={editingLink.url}
                              onChange={(e) =>
                                setEditingLink({
                                  ...editingLink,
                                  url: e.target.value,
                                })
                              }
                              className="h-8"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                type="button"
                                onClick={handleUpdateLink}
                                className="text-xs bg-blue-50 text-blue-600 dark:bg-blue-950/30 dark:text-blue-300 px-2 py-1 rounded hover:bg-blue-100 dark:hover:bg-blue-950/60"
                              >
                                Save
                              </button>
                              <button
                                type="button"
                                onClick={handleCancelEdit}
                                className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded hover:bg-muted/70"
                              >
                                Cancel
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm text-foreground">
                            {link.displayName}
                          </TableCell>
                          <TableCell className="py-0.5 text-sm text-muted-foreground">
                            {link.url}
                          </TableCell>
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
                      <TableCell
                        colSpan={3}
                        className="py-0.5 text-sm text-muted-foreground text-center"
                      >
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
