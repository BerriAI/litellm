import { DataTable } from "./table";
import moment from "moment";
import { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { uiAuditLogsCall, keyListCall } from "../networking";
import { AuditLogEntry, auditLogColumns } from "./columns";
import { Text } from "@tremor/react";
import { Team } from "../key_team_helpers/key_list";
import _ReactDiffViewer from 'react-diff-viewer';
import React from "react";
import Prism from 'prismjs';
import 'prismjs/components/prism-json';
import 'prismjs/themes/prism-tomorrow.css';

const ReactDiffViewer = (_ReactDiffViewer as any).default || _ReactDiffViewer;

interface AuditLogsProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  isActive: boolean;
  premiumUser: boolean;
  allTeams: Team[];
}

// Define AuditLogRowExpansionPanel as a standalone, memoized component
const AuditLogRowExpansionPanel = React.memo(({ rowData }: { rowData: AuditLogEntry }) => {
  const { before_value, updated_values, table_name, action } = rowData;

  // CSS overrides
  const prismStyleOverride = `
    .token.property {
      color: #4A90E2 !important; /* Dark blue for JSON keys */
    }
    .token.string,
    .token.number,
    .token.boolean,
    .token.null {
      color: #578043 !important; /* Darker green for JSON values */
    }
  `;

  // Style overrides for ReactDiffViewer line highlights
  const diffViewerStyles = {
    variables: {
      light: {
        addedBackground: '#D5F5E3', // Darker green for added lines
        removedBackground: '#FADBD8', // Darker red for removed lines
      },
    },
  };
    
  // Custom renderer for the "Expand lines" message
  const renderCodeFoldMessage = (
    numLines: number,
    isFolded: boolean, 
    toggleFold: () => void
  ) => (
    <div
      onClick={toggleFold}
      style={{
        cursor: 'pointer',
        paddingLeft: '35px', 
        fontSize: '0.85em',
        color: '#007bff', 
        textDecoration: 'none', 
      }}
      onMouseEnter={(e) => (e.currentTarget.style.textDecoration = 'underline')}
      onMouseLeave={(e) => (e.currentTarget.style.textDecoration = 'none')}
      title={`Click to ${isFolded ? 'expand' : 'collapse'} ${numLines} lines`}
    >
      {isFolded ? `Expand ${numLines} lines...` : `Collapse ${numLines} lines...`}
    </div>
  );

  // Helper function to recursively mask sk- keys
  const maskSKKeys = (data: any): any => {
    if (typeof data === 'string') {
      // Mask if: starts with sk-, length > 10 (our mask length), and not already in sk-...XXXX format
      if (data.startsWith('sk-') && data.length > 10 && data.substring(3, 6) !== '...') {
        return `${data.substring(0, 3)}...${data.substring(data.length - 4)}`;
      }
      return data;
    }

    if (Array.isArray(data)) {
      return data.map(item => maskSKKeys(item));
    }

    if (typeof data === 'object' && data !== null) {
      const maskedObject: Record<string, any> = {};
      for (const key in data) {
        if (Object.prototype.hasOwnProperty.call(data, key)) {
          maskedObject[key] = maskSKKeys(data[key]);
        }
      }
      return maskedObject;
    }
    
    return data; // Return primitives other than strings, and null, as is
  };

  const getDisplayString = (value: Record<string, any> | string | undefined | null) => {
    if (value === undefined || value === null) return "N/A";
    // If it's already a string (e.g. our "N/A" messages), return it directly.
    // Otherwise, it's an object that needs masking and stringifying.
    if (typeof value === 'string') return value;
    
    const maskedValue = maskSKKeys(value);
    return JSON.stringify(maskedValue, null, 2);
  };

  // Syntax highlighting function for JSON
  const highlightSyntax = (jsonString: string) => {
    // Prism.highlight can throw if language not found, though 'json' should be safe
    try {
      // Check if the string is one of our special "N/A" or "Comment" messages
      if (jsonString === "N/A" || 
          jsonString === "N/A (New Record)" || 
          jsonString === "N/A (Record Deleted)" ||
          jsonString.includes("No fields changed") ||
          jsonString.includes("Updated values were not provided")) {
        // Don't highlight plain text messages, return them as is within a pre for consistency
        return <pre style={{ display: 'inline' }}>{jsonString}</pre>;
      }
      const html = Prism.highlight(jsonString, Prism.languages.json, 'json');
      return <pre style={{ display: 'inline' }} dangerouslySetInnerHTML={{ __html: html }} />;
    } catch (e) {
      console.error("Prism highlighting error:", e);
      // Fallback to plain text if highlighting fails
      return <pre style={{ display: 'inline' }}>{jsonString}</pre>;
    }
  };

  let beforeString = "";
  let updatedString = "";

  if (action === "updated" || action === "rotated") {
    if (before_value && typeof before_value === 'object' && updated_values && typeof updated_values === 'object') {
      // Mask before_value before creating newFullState to ensure diff is on masked data if needed
      const masked_before_value = maskSKKeys(before_value);
      // updated_values might only contain changes, so merge with masked_before_value
      const newFullState = { ...masked_before_value, ...maskSKKeys(updated_values) }; 
      
      const stringifiedOld = getDisplayString(masked_before_value); // getDisplayString will re-mask then stringify
      const stringifiedNew = getDisplayString(newFullState);     // getDisplayString will re-mask then stringify
                                                                 // (maskSKKeys is idempotent for already masked keys)

      if (stringifiedOld === stringifiedNew && JSON.stringify(before_value) === JSON.stringify({ ...before_value, ...updated_values })) {
        beforeString = getDisplayString({ "No fields changed": "N/A" });
        updatedString = getDisplayString({ "No fields changed": "N/A" });
      } else {
        // Pass original unmasked data to getDisplayString, which will handle masking internally
        beforeString = getDisplayString(before_value); 
        updatedString = getDisplayString({ ...before_value, ...updated_values });
      }
    } else if (before_value && !updated_values) { 
        beforeString = getDisplayString(before_value); 
        updatedString = getDisplayString({ "Comment": "Updated values were not provided or were null." });
    } else { 
      beforeString = getDisplayString(before_value); 
      updatedString = getDisplayString(updated_values); 
    }
  } else if (action === "created") {
    beforeString = "N/A (New Record)";
    updatedString = getDisplayString(updated_values); 
  } else if (action === "deleted") {
    beforeString = getDisplayString(before_value);
    updatedString = "N/A (Record Deleted)";
  } else {
     beforeString = getDisplayString(before_value);
     updatedString = getDisplayString(updated_values);
  }
  
  const shouldUseDiffViewer = 
    (action === "updated" || action === "rotated") &&
    beforeString !== updatedString && 
    !(beforeString.includes("No fields changed") && updatedString.includes("No fields changed")) &&
    !(beforeString === "N/A (New Record)" && updatedString === "N/A") && 
    !(beforeString === "N/A" && updatedString === "N/A (Record Deleted)");

  const renderLegacyValue = (rawValue: Record<string, any> | string | undefined | null, isKeyTable: boolean) => {
    let displayValue: any;
    
    if (rawValue === null || rawValue === undefined) {
        return <Text>N/A</Text>;
    }

    // If rawValue is a string that looks like JSON, parse it. Otherwise, use as is.
    // This step should happen BEFORE masking if we expect to mask contents of a JSON string.
    // However, our primary data `before_value`, `updated_values` are objects.
    // `getDisplayString` handles string inputs like "N/A" by returning them.
    // For `renderLegacyValue`, it's safer to assume `rawValue` is either an object or a non-JSON string.
    
    try {
        // If rawValue is a string that represents an object (e.g. "N/A", "No fields changed")
        // it won't parse as JSON and will be handled later.
        // If it's a string that IS JSON, it will be parsed.
        // If it's an object, it remains an object.
        displayValue = (typeof rawValue === 'string' && (rawValue.startsWith('{') || rawValue.startsWith('['))) 
                       ? JSON.parse(rawValue) 
                       : rawValue;
    } catch (e) {
        displayValue = rawValue; 
    }

    // Now, apply masking to the processed displayValue
    const maskedDisplayValue = maskSKKeys(displayValue);

    // If maskedDisplayValue is a string (e.g. "N/A", or a simple string that was passed through maskSKKeys)
    // and it's one of our special messages, render as Text.
    if (typeof maskedDisplayValue === 'string') {
        if (maskedDisplayValue.startsWith("N/A") || maskedDisplayValue.includes("No fields changed") || maskedDisplayValue.includes("Comment")) {
            return <Text>{maskedDisplayValue}</Text>;
        }
         // If it's some other string that's not an object after masking, render it as text
        if (!(maskedDisplayValue.startsWith('{') || maskedDisplayValue.startsWith('['))) {
             return <Text>{maskedDisplayValue}</Text>;
        }
        // If it's a string that still looks like JSON (shouldn't happen if maskSKKeys returned an object), try parsing again
        // This path is unlikely given maskSKKeys behavior.
        try {
            displayValue = JSON.parse(maskedDisplayValue); // Re-assign to displayValue for object checks
        } catch (e) {
            return <Text>{maskedDisplayValue}</Text>; // Render as text if reparsing fails
        }
    } else {
      displayValue = maskedDisplayValue; // It was an object, use the masked version
    }


    if (typeof displayValue !== 'object' || displayValue === null || Object.keys(displayValue).length === 0) {
        return <Text>{typeof displayValue === 'string' ? displayValue : "N/A"}</Text>;
    }

    if (isKeyTable) {
      const changedKeys = Object.keys(displayValue); // use the (potentially) masked displayValue
      const knownKeyFields = ['token', 'spend', 'max_budget'];
      
      const onlyKnownFieldsChanged = changedKeys.every(key => knownKeyFields.includes(key) || displayValue[key] === "N/A");

      if (onlyKnownFieldsChanged && changedKeys.length > 0 && !displayValue["No differing fields detected in 'before' state"] && !displayValue["No differing fields detected in 'updated' state"] && !displayValue["No fields changed"] && !displayValue["Comment"]) {
        return (
          <div>
            {/* Access properties from the (potentially) masked displayValue */}
            {changedKeys.includes('token') && <p><strong>Token:</strong> {displayValue.token || 'N/A'}</p>}
            {changedKeys.includes('spend') && <p><strong>Spend:</strong> {displayValue.spend !== undefined ? `$${Number(displayValue.spend).toFixed(6)}` : 'N/A'}</p>}
            {changedKeys.includes('max_budget') && <p><strong>Max Budget:</strong> {displayValue.max_budget !== undefined ? `$${Number(displayValue.max_budget).toFixed(6)}` : 'N/A'}</p>}
          </div>
        );
      } else {
        const specialMessageKey = Object.keys(displayValue).find(key => 
            key === "No differing fields detected in 'before' state" || 
            key === "No differing fields detected in 'updated' state" || 
            key === "No fields changed" || 
            key === "Comment"
        );
        if (specialMessageKey) {
           return <Text>{displayValue[specialMessageKey]}</Text>;
        }
        // Stringify the (potentially) masked displayValue
        return <pre className="p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-60">{JSON.stringify(displayValue, null, 2)}</pre>;
      }
    }
    // Stringify the (potentially) masked displayValue
    return <pre className="p-2 bg-gray-50 border rounded text-xs overflow-auto max-h-60">{JSON.stringify(displayValue, null, 2)}</pre>;
  };

  return (
    <>
      <style>{prismStyleOverride}</style>
      <div className="-mx-4 p-4 bg-slate-100 border-y border-slate-300">
        {shouldUseDiffViewer ? (
          <ReactDiffViewer
            oldValue={beforeString} // These are already stringified outputs of getDisplayString
            newValue={updatedString} // which internally called maskSKKeys
            splitView={false}
            showDiffOnly={true} 
            hideLineNumbers={false}
            disableWordDiff={true}
            renderContent={highlightSyntax}
            styles={diffViewerStyles}
            codeFoldMessageRenderer={renderCodeFoldMessage}
          />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h4 className="font-semibold mb-2 text-sm text-slate-700">Before Value:</h4>
              {renderLegacyValue(action === "deleted" ? (before_value) : before_value, table_name === "LiteLLM_VerificationToken")}
            </div>
            <div>
              <h4 className="font-semibold mb-2 text-sm text-slate-700">Updated Value:</h4>
              {renderLegacyValue(action === "created" ? (updated_values) : updated_values, table_name === "LiteLLM_VerificationToken")}
            </div>
          </div>
        )}
      </div>
    </>
  );
});
AuditLogRowExpansionPanel.displayName = 'AuditLogRowExpansionPanel'; // Optional: for better debugging

export default function AuditLogs({
  userID,
  userRole,
  token,
  accessToken,
  isActive,
  premiumUser,
  allTeams,
}: AuditLogsProps) {
  const [startTime, setStartTime] = useState<string>(
    moment().subtract(24, "hours").format("YYYY-MM-DDTHH:mm")
  );

  const actionFilterRef = useRef<HTMLDivElement>(null);
  const tableFilterRef = useRef<HTMLDivElement>(null);
  const [clientCurrentPage, setClientCurrentPage] = useState(1);
  const [pageSize] = useState(50);
  const [filters, setFilters] = useState<Record<string, string>>({});
  const [selectedTeamId, setSelectedTeamId] = useState("");
  const [selectedKeyHash, setSelectedKeyHash] = useState("");
  const [objectIdSearch, setObjectIdSearch] = useState("");
  const [selectedActionFilter, setSelectedActionFilter] = useState("all");
  const [selectedTableFilter, setSelectedTableFilter] = useState("all");
  const [actionFilterOpen, setActionFilterOpen] = useState(false);
  const [tableFilterOpen, setTableFilterOpen] = useState(false);

  const allLogsQuery = useQuery<AuditLogEntry[]>({
    queryKey: [
      "all_audit_logs",
      accessToken,
      token,
      userRole,
      userID,
      startTime,
    ],
    queryFn: async () => {
      if (!accessToken || !token || !userRole || !userID) {
        return [];
      }

      const formattedStartTimeStr = moment(startTime).utc().format("YYYY-MM-DD HH:mm:ss");
      const formattedEndTimeStr = moment().utc().format("YYYY-MM-DD HH:mm:ss");

      let accumulatedLogs: AuditLogEntry[] = [];
      let currentPageToFetch = 1;
      let totalPagesFromBackend = 1;
      const backendPageSize = 50;

      do {
        const response = await uiAuditLogsCall(
          accessToken,
          formattedStartTimeStr,
          formattedEndTimeStr,
          currentPageToFetch,
          backendPageSize
        );
        accumulatedLogs = accumulatedLogs.concat(response.audit_logs);
        totalPagesFromBackend = response.total_pages;
        currentPageToFetch++;
      } while (currentPageToFetch <= totalPagesFromBackend);

      return accumulatedLogs;
    },
    enabled: !!accessToken && !!token && !!userRole && !!userID && isActive,
    refetchInterval: 5000,
    refetchIntervalInBackground: true,
  });

  const handleRefresh = () => {
    allLogsQuery.refetch();
  };

  const fetchKeyHashForAlias = useCallback(async (keyAlias: string) => {
    if (!accessToken) return;
    
    try {
      const response = await keyListCall(
        accessToken,
        null,
        null,
        keyAlias,
        null,
        null,
        1,
        10
      );

      const selectedKey = response.keys.find(
        (key: any) => key.key_alias === keyAlias
      );

      if (selectedKey) {
        setSelectedKeyHash(selectedKey.token);
      } else {
        setSelectedKeyHash("");
      }
    } catch (error) {
      console.error("Error fetching key hash for alias:", error);
      setSelectedKeyHash("");
    }
  }, [accessToken]);

  useEffect(() => {
    if(!accessToken) return;

    let teamIdChanged = false;
    let keyHashChanged = false;

    if (filters['Team ID']) {
      if (selectedTeamId !== filters['Team ID']) {
        setSelectedTeamId(filters['Team ID']);
        teamIdChanged = true;
      }
    } else {
      if (selectedTeamId !== "") {
        setSelectedTeamId("");
        teamIdChanged = true;
      }
    }
    
    if (filters['Key Hash']) {
      if (selectedKeyHash !== filters['Key Hash']) {
        setSelectedKeyHash(filters['Key Hash']);
        keyHashChanged = true;
      }
    } else if (filters['Key Alias']) {
      fetchKeyHashForAlias(filters['Key Alias']);
    } else {
      if (selectedKeyHash !== "") {
        setSelectedKeyHash("");
        keyHashChanged = true;
      }
    }

    if (teamIdChanged || keyHashChanged) {
        setClientCurrentPage(1);
    }
  }, [filters, accessToken, fetchKeyHashForAlias, selectedTeamId, selectedKeyHash]);

  useEffect(() => {
    setClientCurrentPage(1);
  }, [selectedTeamId, selectedKeyHash, startTime, objectIdSearch, selectedActionFilter, selectedTableFilter]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        actionFilterRef.current &&
        !actionFilterRef.current.contains(event.target as Node)
      ) {
        setActionFilterOpen(false);
      }
      if (
        tableFilterRef.current &&
        !tableFilterRef.current.contains(event.target as Node)
      ) {
        setTableFilterOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const completeFilteredLogs = useMemo(() => {
    if (!allLogsQuery.data) return [];
    return allLogsQuery.data.filter(log => {
      let matchesTeam = true;
      let matchesKey = true;
      let matchesObjectId = true;
      let matchesAction = true;
      let matchesTable = true;

      if (selectedTeamId) {
        const beforeTeamId = typeof log.before_value === 'string' ? JSON.parse(log.before_value)?.team_id : log.before_value?.team_id;
        const updatedTeamId = typeof log.updated_values === 'string' ? JSON.parse(log.updated_values)?.team_id : log.updated_values?.team_id;
        matchesTeam = beforeTeamId === selectedTeamId || updatedTeamId === selectedTeamId;
      }

      if (selectedKeyHash) {
        try {
          const beforeBody = typeof log.before_value === 'string' ? JSON.parse(log.before_value) : log.before_value;
          const updatedBody = typeof log.updated_values === 'string' ? JSON.parse(log.updated_values) : log.updated_values;
          
          const beforeKey = beforeBody?.token;
          const updatedKey = updatedBody?.token;

          matchesKey = (typeof beforeKey === 'string' && beforeKey.includes(selectedKeyHash)) ||
                      (typeof updatedKey === 'string' && updatedKey.includes(selectedKeyHash));
        } catch (e) {
          matchesKey = false;
        }
      }

      if (objectIdSearch) {
        matchesObjectId = log.object_id?.toLowerCase().includes(objectIdSearch.toLowerCase());
      }

      if (selectedActionFilter !== "all") {
        matchesAction = log.action?.toLowerCase() === selectedActionFilter.toLowerCase();
      }

      if (selectedTableFilter !== "all") {
        let tableMatchName = "";
        switch(selectedTableFilter) {
          case "keys": tableMatchName = "litellm_verificationtoken"; break;
          case "teams": tableMatchName = "litellm_teamtable"; break;
          case "users": tableMatchName = "litellm_usertable"; break;
          // Add other direct table names if needed, or rely on a more generic match
          default: tableMatchName = selectedTableFilter; // Should not happen with current UI options
        }
        matchesTable = log.table_name?.toLowerCase() === tableMatchName;
      }

      return matchesTeam && matchesKey && matchesObjectId && matchesAction && matchesTable;
    });
  }, [allLogsQuery.data, selectedTeamId, selectedKeyHash, objectIdSearch, selectedActionFilter, selectedTableFilter]);

  const totalFilteredItems = completeFilteredLogs.length;
  const totalFilteredPages = Math.ceil(totalFilteredItems / pageSize) || 1;

  const paginatedViewOfFilteredLogs = useMemo(() => {
    const start = (clientCurrentPage - 1) * pageSize;
    const end = start + pageSize;
    return completeFilteredLogs.slice(start, end);
  }, [completeFilteredLogs, clientCurrentPage, pageSize]);

  // renderSubComponent now simply instantiates the memoized component
  const renderSubComponent = useCallback(({ row }: { row: any }) => {
    return <AuditLogRowExpansionPanel rowData={row.original as AuditLogEntry} />;
  }, []); // Empty dependency array means this callback itself is stable

  if (!premiumUser) {
    return (
      <div>
        <Text>This is a LiteLLM Enterprise feature, and requires a valid key to use. Get a trial key <a href="https://litellm.ai/pricing" target="_blank" rel="noopener noreferrer">here</a>.</Text>
      </div>
    );
  }

  const currentDisplayItemsStart = totalFilteredItems > 0 ? (clientCurrentPage - 1) * pageSize + 1 : 0;
  const currentDisplayItemsEnd = Math.min(clientCurrentPage * pageSize, totalFilteredItems);

  return (
    <>
      <div className="flex items-center justify-between mb-4">
        
      </div>
      {/* <FilterComponent options={auditLogFilterOptions} onApplyFilters={handleFilterChange} onResetFilters={handleFilterReset} /> */}
      <div className="bg-white rounded-lg shadow">
        <div className="border-b px-6 py-4">
        <h1 className="text-xl font-semibold py-4">
              Audit Logs
            </h1>
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between space-y-4 md:space-y-0">
            
            <div className="flex flex-wrap items-center gap-3">
            
              <div className="flex items-center gap-2">
                <div className="flex items-center">
                  <input
                    type="text"
                    placeholder="Search by Object ID..."
                    value={objectIdSearch}
                    onChange={(e) => setObjectIdSearch(e.target.value)}
                    className="px-3 py-2 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
                
                <button
                  onClick={handleRefresh}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                  title="Refresh data"
                >
                  <svg
                    className={`w-4 h-4 ${allLogsQuery.isFetching ? 'animate-spin' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  <span>Refresh</span>
                </button>
              </div>

            </div>

            <div className="flex items-center space-x-4">
              {/* Custom Action Filter Dropdown */}
              <div className="relative" ref={actionFilterRef}>
                <label htmlFor="actionFilterDisplay" className="mr-2 text-sm font-medium text-gray-700 sr-only">Action:</label>
                <button
                  id="actionFilterDisplay"
                  onClick={() => setActionFilterOpen(!actionFilterOpen)}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 bg-white w-40 text-left justify-between"
                >
                  <span>
                    {selectedActionFilter === "all" && "All Actions"}
                    {selectedActionFilter === "created" && "Created"}
                    {selectedActionFilter === "updated" && "Updated"}
                    {selectedActionFilter === "deleted" && "Deleted"}
                    {selectedActionFilter === "rotated" && "Rotated"}
                  </span>
                  <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
                {actionFilterOpen && (
                  <div className="absolute left-0 mt-2 w-40 bg-white rounded-lg shadow-lg border p-1 z-50">
                    <div className="space-y-1">
                      {[
                        { label: "All Actions", value: "all" },
                        { label: "Created", value: "created" },
                        { label: "Updated", value: "updated" },
                        { label: "Deleted", value: "deleted" },
                        { label: "Rotated", value: "rotated" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                            selectedActionFilter === option.value ? 'bg-blue-50 text-blue-600 font-medium' : 'font-normal'
                          }`}
                          onClick={() => {
                            setSelectedActionFilter(option.value);
                            setActionFilterOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Custom Table Filter Dropdown */}
              <div className="relative" ref={tableFilterRef}>
                <label htmlFor="tableFilterDisplay" className="mr-2 text-sm font-medium text-gray-700 sr-only">Table:</label>
                <button
                  id="tableFilterDisplay"
                  onClick={() => setTableFilterOpen(!tableFilterOpen)}
                  className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 bg-white w-40 text-left justify-between"
                >
                  <span>
                    {selectedTableFilter === "all" && "All Tables"}
                    {selectedTableFilter === "keys" && "Keys"}
                    {selectedTableFilter === "teams" && "Teams"}
                    {selectedTableFilter === "users" && "Users"}
                  </span>
                  <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
                {tableFilterOpen && (
                  <div className="absolute left-0 mt-2 w-40 bg-white rounded-lg shadow-lg border p-1 z-50">
                    <div className="space-y-1">
                      {[
                        { label: "All Tables", value: "all" },
                        { label: "Keys", value: "keys" },
                        { label: "Teams", value: "teams" },
                        { label: "Users", value: "users" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          className={`w-full px-3 py-2 text-left text-sm hover:bg-gray-50 rounded-md ${
                            selectedTableFilter === option.value ? 'bg-blue-50 text-blue-600 font-medium' : 'font-normal'
                          }`}
                          onClick={() => {
                            setSelectedTableFilter(option.value);
                            setTableFilterOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              
              <span className="text-sm text-gray-700">
                Showing{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : currentDisplayItemsStart}{" "}
                -{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : currentDisplayItemsEnd}{" "}
                of{" "}
                {allLogsQuery.isLoading
                  ? "..."
                  : totalFilteredItems}{" "}
                results
              </span>
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-700">
                  Page {allLogsQuery.isLoading ? "..." : clientCurrentPage} of{" "}
                  {allLogsQuery.isLoading
                    ? "..."
                    : totalFilteredPages}
                </span>
                <button
                  onClick={() =>
                    setClientCurrentPage((p) => Math.max(1, p - 1))
                  }
                  disabled={allLogsQuery.isLoading || clientCurrentPage === 1}
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Previous
                </button>
                <button
                  onClick={() =>
                    setClientCurrentPage((p) =>
                      Math.min(
                        totalFilteredPages,
                        p + 1,
                      ),
                    )
                  }
                  disabled={
                    allLogsQuery.isLoading ||
                    clientCurrentPage === totalFilteredPages
                  }
                  className="px-3 py-1 text-sm border rounded-md hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>
        <DataTable
          columns={auditLogColumns}
          data={paginatedViewOfFilteredLogs}
          renderSubComponent={renderSubComponent}
          getRowCanExpand={() => true}
        />
      </div>
    </>
  );
}
