import TableIconActionButton from "@/components/common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { isAdminRole } from "@/utils/roles";
import { ChevronDownIcon, ChevronRightIcon, ExternalLinkIcon, PlusCircleIcon } from "@heroicons/react/outline";
import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import Link from "next/link";
import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl, getPublicModelHubInfo, getUiPageUrl, updateUsefulLinksCall } from "../networking";
import { toUsefulLinksRequest } from "./usefulLinksMapper";
import { createUsefulLinksSchemas, type UsefulLink } from "./usefulLinksSchema";

interface UsefulLinksManagementProps {
  accessToken: string | null;
  userRole: string | null;
}

const emptyLink = { url: "", displayName: "" };

const UsefulLinksManagement: React.FC<UsefulLinksManagementProps> = ({ accessToken, userRole }) => {
  const { t } = useTranslation();
  const [links, setLinks] = useState<UsefulLink[]>([]);
  const { addSchema, editSchema } = createUsefulLinksSchemas(links);
  const addForm = useZodForm(addSchema, { defaultValues: emptyLink });
  const editForm = useZodForm(editSchema, { defaultValues: { link: null } });
  const newLink = addForm.watch();
  const editingLink = editForm.watch("link");
  const [isExpanded, setIsExpanded] = useState(true);
  const [isRearranging, setIsRearranging] = useState(false);
  const [originalLinksOrder, setOriginalLinksOrder] = useState<UsefulLink[]>([]);

  const fetchUsefulLinks = async () => {
    if (!accessToken) return;

    try {
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
                url: value.url,
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
    }
  };

  useEffect(() => {
    fetchUsefulLinks();
  }, [accessToken]);

  // Check if user is admin
  if (!isAdminRole(userRole || "")) {
    return null;
  }

  const saveLinksToBackend = async (updatedLinks: UsefulLink[]) => {
    if (!accessToken) return false;

    try {
      const { useful_links } = toUsefulLinksRequest(updatedLinks);
      await updateUsefulLinksCall(accessToken, useful_links);

      return true;
    } catch (error) {
      console.error("Error saving links:", error);
      toast.fromError(t("aiHub.usefulLinks.saveFailed", { error }));
      return false;
    }
  };

  const showDraftError = (message: string | undefined) => {
    if (message === "invalidUrl" || message === "duplicateName") {
      toast.fromError(t(`aiHub.usefulLinks.${message}`));
    }
  };

  const handleAddLink = async () => {
    const result = addSchema.safeParse(addForm.getValues());
    if (!result.success) {
      showDraftError(result.error.issues[0]?.message);
      return;
    }
    const { displayName, url } = result.data;
    const newLinkObj: UsefulLink = {
      id: `${Date.now()}-${displayName}`,
      displayName,
      url,
    };

    const updatedLinks = [...links, newLinkObj];

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      addForm.reset(emptyLink);
      toast.success(t("aiHub.usefulLinks.added"));
    }
  };

  const handleEditLink = (link: UsefulLink) => {
    editForm.setValue("link", link);
  };

  const handleUpdateLink = async () => {
    if (!editingLink) return;

    const result = editSchema.safeParse(editForm.getValues());
    if (!result.success) {
      showDraftError(result.error.issues[0]?.message);
      return;
    }

    const editedLink = result.data.link;
    if (!editedLink) return;
    const updatedLinks = links.map((link) => (link.id === editedLink.id ? editedLink : link));

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      handleCancelEdit();
      toast.success(t("aiHub.usefulLinks.updated"));
    }
  };

  const handleCancelEdit = () => {
    editForm.setValue("link", null);
  };

  const deleteLink = async (linkId: string) => {
    const updatedLinks = links.filter((link) => link.id !== linkId);

    if (await saveLinksToBackend(updatedLinks)) {
      setLinks(updatedLinks);
      toast.success(t("aiHub.usefulLinks.deleted"));
    }
  };

  const setCurrentLink = (url: string) => {
    window.open(url, "_blank");
  };

  const handleStartRearranging = () => {
    if (editingLink) {
      handleCancelEdit();
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
      toast.success(t("aiHub.usefulLinks.orderSaved"));
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
    <Card className="mb-6 px-6">
      <div className="flex items-center justify-between cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex flex-col">
          <h3 className="mb-0 text-lg font-semibold">{t("aiHub.usefulLinks.title")}</h3>
          <p className="text-sm text-muted-foreground">{t("aiHub.usefulLinks.description")}</p>
        </div>
        <div className="flex items-center">
          {isExpanded ? (
            <ChevronDownIcon className="w-5 h-5 text-muted-foreground" />
          ) : (
            <ChevronRightIcon className="w-5 h-5 text-muted-foreground" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4">
          <div className="mb-6">
            <p className="text-sm font-medium text-foreground mb-2">{t("aiHub.usefulLinks.addForm.title")}</p>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-muted-foreground mb-1">
                  {t("aiHub.usefulLinks.addForm.displayName")}
                </label>
                <input
                  type="text"
                  value={newLink.displayName}
                  onChange={(e) => addForm.setValue("displayName", e.target.value)}
                  placeholder={t("aiHub.usefulLinks.addForm.displayNamePlaceholder")}
                  className="w-full px-3 py-2 border border-border rounded-md text-sm"
                />
              </div>
              <div>
                <label className="block text-xs text-muted-foreground mb-1">{t("aiHub.usefulLinks.addForm.url")}</label>
                <input
                  type="text"
                  value={newLink.url}
                  onChange={(e) => addForm.setValue("url", e.target.value)}
                  placeholder="https://example.com"
                  className="w-full px-3 py-2 border border-border rounded-md text-sm"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={handleAddLink}
                  disabled={!newLink.url || !newLink.displayName}
                  className={`flex items-center px-4 py-2 rounded-md text-sm ${!newLink.url || !newLink.displayName ? "bg-border text-muted-foreground cursor-not-allowed" : "bg-success text-success-foreground hover:bg-success/80"}`}
                >
                  <PlusCircleIcon className="w-4 h-4 mr-1" />
                  {t("aiHub.usefulLinks.addForm.submit")}
                </button>
              </div>
            </div>
          </div>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-medium text-foreground">{t("aiHub.usefulLinks.manageExisting")}</p>
            <div className="flex items-center space-x-2">
              <Link
                href={getUiPageUrl(`${getProxyBaseUrl()}/ui/model_hub_table`)}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs bg-info/10 text-info px-3 py-1.5 rounded-sm hover:bg-info/15 flex items-center"
                title={t("aiHub.usefulLinks.openPublicHubTitle")}
              >
                {t("aiHub.usefulLinks.publicHub")}
                <ExternalLinkIcon className="w-4 h-4 ml-1" />
              </Link>
              {!isRearranging ? (
                <button
                  onClick={handleStartRearranging}
                  className="text-xs bg-purple-50 text-purple-600 px-3 py-1.5 rounded-sm hover:bg-purple-100 flex items-center dark:bg-purple-950 dark:text-purple-300 dark:hover:bg-purple-900"
                >
                  {t("aiHub.usefulLinks.rearrange")}
                </button>
              ) : (
                <div className="flex space-x-2">
                  <button
                    onClick={handleSaveRearranging}
                    className="text-xs bg-success text-success-foreground px-3 py-1.5 rounded-sm hover:bg-success/80"
                  >
                    {t("aiHub.usefulLinks.saveOrder")}
                  </button>
                  <button
                    onClick={handleCancelRearranging}
                    className="text-xs bg-muted text-muted-foreground px-3 py-1.5 rounded-sm hover:bg-accent"
                  >
                    {t("aiHub.usefulLinks.cancel")}
                  </button>
                </div>
              )}
            </div>
          </div>
          <div className="rounded-lg custom-border relative">
            <div className="overflow-x-auto">
              <Table className="[&_td]:py-0.5 [&_th]:py-1">
                <TableHeader>
                  <TableRow>
                    <TableHead className="py-1 h-8">{t("aiHub.usefulLinks.table.displayName")}</TableHead>
                    <TableHead className="py-1 h-8">{t("aiHub.usefulLinks.table.url")}</TableHead>
                    <TableHead className="py-1 h-8">{t("aiHub.usefulLinks.table.actions")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {links.map((link, index) => (
                    <TableRow key={link.id} className="h-8">
                      {editingLink && editingLink.id === link.id ? (
                        <>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingLink.displayName}
                              onChange={(e) => editForm.setValue("link", { ...editingLink, displayName: e.target.value })}
                              className="w-full px-2 py-1 border border-border rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5">
                            <input
                              type="text"
                              value={editingLink.url as string}
                              onChange={(e) => editForm.setValue("link", { ...editingLink, url: e.target.value })}
                              className="w-full px-2 py-1 border border-border rounded-md text-sm"
                            />
                          </TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            <div className="flex space-x-2">
                              <button
                                onClick={handleUpdateLink}
                                className="text-xs bg-info/10 text-info px-2 py-1 rounded-sm hover:bg-info/15"
                              >
                                {t("aiHub.usefulLinks.save")}
                              </button>
                              <button
                                onClick={handleCancelEdit}
                                className="text-xs bg-muted text-muted-foreground px-2 py-1 rounded-sm hover:bg-accent"
                              >
                                {t("aiHub.usefulLinks.cancel")}
                              </button>
                            </div>
                          </TableCell>
                        </>
                      ) : (
                        <>
                          <TableCell className="py-0.5 text-sm text-foreground">{link.displayName}</TableCell>
                          <TableCell className="py-0.5 text-sm text-muted-foreground">{link.url as string}</TableCell>
                          <TableCell className="py-0.5 whitespace-nowrap">
                            {isRearranging ? (
                              <div className="flex space-x-2">
                                <TableIconActionButton
                                  variant="Up"
                                  onClick={() => handleMoveUp(index)}
                                  tooltipText={t("aiHub.usefulLinks.moveUp")}
                                  disabled={index === 0}
                                  disabledTooltipText={t("aiHub.usefulLinks.alreadyTop")}
                                  dataTestId={`move-up-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Down"
                                  onClick={() => handleMoveDown(index)}
                                  tooltipText={t("aiHub.usefulLinks.moveDown")}
                                  disabled={index === links.length - 1}
                                  disabledTooltipText={t("aiHub.usefulLinks.alreadyBottom")}
                                  dataTestId={`move-down-${link.id}`}
                                />
                              </div>
                            ) : (
                              <div className="flex space-x-2">
                                <TableIconActionButton
                                  variant="Open"
                                  onClick={() => setCurrentLink(link.url as string)}
                                  tooltipText={t("aiHub.usefulLinks.openLink")}
                                  dataTestId={`open-link-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Edit"
                                  onClick={() => handleEditLink(link)}
                                  tooltipText={t("aiHub.usefulLinks.editLink")}
                                  dataTestId={`edit-link-${link.id}`}
                                />
                                <TableIconActionButton
                                  variant="Delete"
                                  onClick={() => deleteLink(link.id)}
                                  tooltipText={t("aiHub.usefulLinks.deleteLink")}
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
                      <TableCell colSpan={3} className="py-0.5 text-sm text-muted-foreground text-center">
                        {t("aiHub.usefulLinks.empty")}
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
