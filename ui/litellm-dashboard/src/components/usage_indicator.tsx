import { useState, useEffect } from "react"
import { Badge } from "@tremor/react"
import { AlertTriangle, Users, TrendingUp, Loader2, ChevronDown, ChevronUp, Minus } from "lucide-react"
import { getRemainingUsers } from "./networking"

// Simple utility function to combine class names
const cn = (...classes: (string | boolean | undefined)[]) => {
  return classes.filter(Boolean).join(' ')
}

interface UsageIndicatorProps {
  accessToken: string | null
  width: number
}

interface UsageData {
  total_users: number | null
  total_users_used: number
  total_users_remaining: number | null
}

export default function UsageIndicator({accessToken, width = 220}: UsageIndicatorProps) {
  const position = "bottom-left"
  const [isExpanded, setIsExpanded] = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [data, setData] = useState<UsageData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken) return
      
      setIsLoading(true)
      setError(null)
      
      try {
        const result = await getRemainingUsers(accessToken)
        setData(result)
      } catch (err) {
        console.error('Failed to fetch usage data:', err)
        setError('Failed to load usage data')
      } finally {
        setIsLoading(false)
      }
    }
    
    fetchData()
  }, [accessToken])

  // Calculate derived values from data
  const getUsageMetrics = (data: UsageData | null) => {
    if (!data) {
      return {
        isOverLimit: false,
        isNearLimit: false,
        usagePercentage: 0
      }
    }

    const isOverLimit = data.total_users_remaining ? data.total_users_remaining <= 0 : false
    const isNearLimit = data.total_users_remaining ? data.total_users_remaining <= 5 && data.total_users_remaining > 0 : false
    const usagePercentage = data.total_users ? (data.total_users_used / data.total_users) * 100 : 0

    return { isOverLimit, isNearLimit, usagePercentage }
  }

  const { isOverLimit, isNearLimit, usagePercentage } = getUsageMetrics(data)

  const getStatusColor = () => {
    if (isOverLimit) return "red"
    if (isNearLimit) return "yellow"
    return "green"
  }

  const getStatusText = () => {
    if (isOverLimit) return "Over Limit"
    if (isNearLimit) return "Near Limit"
    return "Active"
  }

  const getStatusIcon = () => {
    if (isOverLimit) return <AlertTriangle className="h-3 w-3" />
    if (isNearLimit) return <TrendingUp className="h-3 w-3" />
    return null
  }

  // Minimized view - just a small restore button
  const MinimizedView = () => {
    const hasIssues = isOverLimit || isNearLimit
    
    return (
      <div 
        className="px-3 py-1"
        style={{ maxWidth: `${width}px` }}
      >
        <button
          onClick={() => setIsMinimized(false)}
          className={cn(
            "flex items-center gap-2 text-xs text-gray-400 hover:text-gray-600 transition-colors p-1 rounded w-full",
            hasIssues && isOverLimit && "text-red-400 hover:text-red-600",
            hasIssues && isNearLimit && "text-yellow-500 hover:text-yellow-700"
          )}
          title="Show usage details"
        >
          <Users className="h-3 w-3 flex-shrink-0" />
          {hasIssues && <span className="flex-shrink-0">{getStatusIcon()}</span>}
          <span className="truncate">{data ? `${data.total_users_used}/${data.total_users}` : "Usage"}</span>
        </button>
      </div>
    )
  }

  // Sidebar/nav style component
  const NavStyleView = () => {
    if (isMinimized) {
      return <MinimizedView />
    }

    if (isLoading) {
      return (
        <div 
          className="flex items-center gap-3 px-3 py-2 text-gray-500"
          style={{ maxWidth: `${width}px` }}
        >
          <Loader2 className="h-4 w-4 animate-spin flex-shrink-0" />
          <span className="text-sm truncate">Loading...</span>
        </div>
      )
    }

    if (error || !data) {
      return (
        <div 
          className="flex items-center justify-between gap-3 px-3 py-2 text-gray-400 group"
          style={{ maxWidth: `${width}px` }}
        >
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <Users className="h-4 w-4 flex-shrink-0" />
            <span className="text-sm truncate">{error || "No data"}</span>
          </div>
          <button
            onClick={() => setIsMinimized(true)}
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-gray-100 rounded transition-all flex-shrink-0"
            title="Minimize"
          >
            <Minus className="h-3 w-3" />
          </button>
        </div>
      )
    }

    return (
      <div 
        className="px-3 py-2 group"
        style={{ maxWidth: `${width}px` }}
      >
        {/* Main nav item style */}
        <div className="flex items-center justify-between">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className={cn(
              "flex items-center gap-3 text-left hover:bg-gray-50 rounded-md px-0 py-1 transition-colors flex-1 min-w-0",
              isOverLimit && "text-red-600",
              isNearLimit && "text-yellow-600"
            )}
          >
            <Users className="h-4 w-4 flex-shrink-0" />
            <span className="text-sm font-medium truncate">Usage Status</span>
            {(isOverLimit || isNearLimit) && (
              <Badge color={getStatusColor()} className="text-xs px-1.5 py-0.5 flex-shrink-0">
                {getStatusIcon()}
              </Badge>
            )}
            {isExpanded ? (
              <ChevronUp className="h-3 w-3 text-gray-400 ml-auto flex-shrink-0" />
            ) : (
              <ChevronDown className="h-3 w-3 text-gray-400 ml-auto flex-shrink-0" />
            )}
          </button>
          
          {/* Minimize button */}
          <button
            onClick={() => setIsMinimized(true)}
            className="opacity-0 group-hover:opacity-100 p-0.5 hover:bg-gray-100 rounded transition-all ml-1 flex-shrink-0"
            title="Minimize"
          >
            <Minus className="h-3 w-3 text-gray-400" />
          </button>
        </div>

        {/* Expanded details - simple and compact */}
        {isExpanded && (
          <div className="mt-2 pl-7 text-xs text-gray-600">
            <div className="mb-1">
              <span className="font-medium">{data.total_users_used}/{data.total_users}</span>
              <span className="text-gray-500"> users</span>
            </div>
            
            {/* Simple progress bar */}
            <div className="w-full bg-gray-200 rounded-full h-1 mb-1">
              <div
                className={cn(
                  "h-1 rounded-full transition-all duration-300",
                  isOverLimit && "bg-red-500",
                  isNearLimit && "bg-yellow-500",
                  !isOverLimit && !isNearLimit && "bg-green-500",
                )}
                style={{ width: `${Math.min(usagePercentage, 100)}%` }}
              />
            </div>

            {(isOverLimit || isNearLimit) && (
              <div className={cn(
                "flex items-center gap-1 text-xs",
                isOverLimit && "text-red-600",
                isNearLimit && "text-yellow-600"
              )}>
                {getStatusIcon()}
                <span className="truncate">{getStatusText()}</span>
              </div>
            )}
          </div>
        )}
      </div>
    )
  }

  // Optimized CardStyleView for 220px width
  const CardStyleView = () => {
    if (isMinimized) {
      const hasIssues = isOverLimit || isNearLimit
      return (
        <button
          onClick={() => setIsMinimized(false)}
          className={cn(
            "bg-white border border-gray-200 rounded-lg shadow-sm p-3 hover:shadow-md transition-all w-full",
            hasIssues && isOverLimit && "border-red-200 bg-red-50",
            hasIssues && isNearLimit && "border-yellow-200 bg-yellow-50"
          )}
          title="Show usage details"
        >
          <div className="flex items-center gap-2">
            <Users className="h-4 w-4 flex-shrink-0" />
            {hasIssues && <span className="flex-shrink-0">{getStatusIcon()}</span>}
            {data && (
              <span className="text-sm font-medium truncate">
                {data.total_users_used}/{data.total_users}
              </span>
            )}
          </div>
        </button>
      )
    }

    if (isLoading) {
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4 w-full">
          <div className="flex items-center justify-center gap-2 py-2">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="text-sm text-gray-500 truncate">Loading...</span>
          </div>
        </div>
      )
    }

    if (error || !data) {
      return (
        <div className="bg-white border border-gray-200 rounded-lg shadow-sm p-4 group w-full">
          <div className="flex items-center justify-between gap-2">
            <div className="flex-1 min-w-0">
              <span className="text-sm text-gray-500 truncate block">
                {error || "No data"}
              </span>
            </div>
            <button
              onClick={() => setIsMinimized(true)}
              className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 rounded transition-all flex-shrink-0"
              title="Minimize"
            >
              <Minus className="h-3 w-3 text-gray-400" />
            </button>
          </div>
        </div>
      )
    }

    return (
      <div 
        className={cn(
          "bg-white border rounded-lg shadow-sm p-3 transition-all duration-200 group w-full",
          isOverLimit && "border-red-200 bg-red-50",
          isNearLimit && "border-yellow-200 bg-yellow-50",
        )}
      >
        {/* Header with title and minimize button */}
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <Users className="h-4 w-4 flex-shrink-0" />
            <span className="font-medium text-sm truncate">Usage</span>
            {(isOverLimit || isNearLimit) && (
              <Badge color={getStatusColor()} className="text-xs px-1.5 py-0.5 flex-shrink-0">
                {getStatusText()}
              </Badge>
            )}
          </div>
          <button
            onClick={() => setIsMinimized(true)}
            className="opacity-0 group-hover:opacity-100 p-1 hover:bg-gray-100 rounded transition-all flex-shrink-0"
            title="Minimize"
          >
            <Minus className="h-3 w-3 text-gray-400" />
          </button>
        </div>

        {/* Compact stats optimized for 220px */}
        <div className="space-y-2 text-sm">
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-xs">Used:</span>
            <span className="font-medium text-right">{data.total_users_used}/{data.total_users}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-xs">Remaining:</span>
            <span className={cn(
              "font-medium text-right",
              isOverLimit && "text-red-600",
              isNearLimit && "text-yellow-600"
            )}>
              {data.total_users_remaining}
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-600 text-xs">Usage:</span>
            <span className="font-medium text-right">{Math.round(usagePercentage)}%</span>
          </div>

          {/* Progress bar */}
          <div className="w-full bg-gray-200 rounded-full h-2 mt-3">
            <div
              className={cn(
                "h-2 rounded-full transition-all duration-300",
                isOverLimit && "bg-red-500",
                isNearLimit && "bg-yellow-500",
                !isOverLimit && !isNearLimit && "bg-green-500",
              )}
              style={{ width: `${Math.min(usagePercentage, 100)}%` }}
            />
          </div>
        </div>
      </div>
    )
  }

  // Don't render anything if no access token or if total_users is null
  if (!accessToken || data?.total_users === null) {
    return null
  }

  // Fixed positioning with proper spacing from edges
  return (
    <div 
      className="fixed bottom-4 left-4 z-50"
      style={{ width: `${Math.min(width, 220)}px` }}
    >
      <CardStyleView />
    </div>
  )
}
