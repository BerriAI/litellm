"use client"

import { useState } from "react"
import { Card, Badge, Button } from "@tremor/react"
import { AlertTriangle, Users, TrendingUp, X, ChevronUp, ChevronDown } from "lucide-react"

// Simple utility function to combine class names
const cn = (...classes: (string | boolean | undefined)[]) => {
  return classes.filter(Boolean).join(' ')
}

const usageData = {
    total_users: 10,
    total_users_used: 255,
    total_users_remaining: -245,
}

interface UsageData {
  total_users: number
  total_users_used: number
  total_users_remaining: number
}

interface UsageIndicatorProps {
//   data: UsageData
  position?: "bottom-left" | "top-right" | "sidebar" | "banner"
  className?: string
}

export default function UsageIndicator({ position = "bottom-left", className }: UsageIndicatorProps) {
  const [isMinimized, setIsMinimized] = useState(false)

  const data = usageData

  const usagePercentage = (data.total_users_used / data.total_users) * 100
  const isOverLimit = data.total_users_remaining < 0
  const isNearLimit = data.total_users_remaining <= data.total_users * 0.1 && !isOverLimit

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

  const positionClasses = {
    "bottom-left": "fixed bottom-4 left-4 z-50",
    "top-right": "fixed top-4 right-4 z-50",
    sidebar: "w-full",
    banner: "w-full",
  }

  const MinimizedView = () => (
    <Card
      className={cn(
        "cursor-pointer transition-all duration-200 hover:shadow-md p-3",
        isOverLimit && "border-red-200 bg-red-50",
        isNearLimit && "border-yellow-200 bg-yellow-50",
      )}
      onClick={() => setIsMinimized(false)}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Users className="h-4 w-4" />
          <span className="text-sm font-medium">
            {data.total_users_used}/{data.total_users}
          </span>
          <Badge color={getStatusColor()} className="text-xs inline-flex items-center">
            {getStatusText()}
          </Badge>
        </div>
        <ChevronUp className="h-3 w-3 text-gray-400 flex-shrink-0" />
      </div>
    </Card>
  )

  const FullView = () => (
    <Card
      className={cn(
        "transition-all duration-200 p-4",
        isOverLimit && "border-red-200 bg-red-50",
        isNearLimit && "border-yellow-200 bg-yellow-50",
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <Users className="h-4 w-4" />
          <span className="font-medium text-sm">Usage Status</span>
        </div>
        <div className="flex items-center gap-1">
          <Button variant="light" size="xs" className="h-6 w-6 p-0" onClick={() => setIsMinimized(true)}>
            <ChevronDown className="h-3 w-3" />
          </Button>
          <Button variant="light" size="xs" className="h-6 w-6 p-0" onClick={() => setIsMinimized(true)}>
            <X className="h-3 w-3" />
          </Button>
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Users Used</span>
          <span className="font-medium">{data.total_users_used.toLocaleString()}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Total Limit</span>
          <span className="font-medium">{data.total_users.toLocaleString()}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-600">Remaining</span>
          <span className={cn("font-medium", isOverLimit && "text-red-600", isNearLimit && "text-yellow-600")}>
            {data.total_users_remaining.toLocaleString()}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-gray-600">
            <span>Usage</span>
            <span>{Math.round(usagePercentage)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
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

        <Badge color={getStatusColor()} className="w-full justify-center">
          <div className="flex items-center justify-center">
            {isOverLimit && <AlertTriangle className="h-3 w-3 mr-1" />}
            {isNearLimit && <TrendingUp className="h-3 w-3 mr-1" />}
            <span>{getStatusText()}</span>
          </div>
        </Badge>
      </div>
    </Card>
  )

  if (position === "banner") {
    return (
      <div className={cn("w-full", className)}>
        <FullView />
      </div>
    )
  }

  return (
    <div className={cn(positionClasses[position], className)}>{isMinimized ? <MinimizedView /> : <FullView />}</div>
  )
}
