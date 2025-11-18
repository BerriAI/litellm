"use client";

import React, { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Zap,
  Users,
  BarChart3,
  Shield,
  Rocket,
  Eye,
  Plus,
  Activity,
  TrendingUp,
  Globe,
  Cpu,
  Key,
  Settings,
  PlayCircle,
  Layers
} from "lucide-react";

interface SkypieaDashboardProps {
  userRole: string;
  userID: string | null;
  accessToken: string | null;
  onNavigate: (page: string) => void;
}

export default function SkypieaDashboard({ userRole, userID, accessToken, onNavigate }: SkypieaDashboardProps) {
  const [metrics, setMetrics] = useState({
    totalCalls: 0,
    activeModels: 0,
    uptime: 0,
    costSavings: 0
  });

  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Simulate loading metrics
    const timer = setTimeout(() => {
      setMetrics({
        totalCalls: 12847,
        activeModels: 45,
        uptime: 99.8,
        costSavings: 2340.50
      });
      setIsLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const quickActions = [
    {
      title: "Create API Key",
      description: "Generate new access keys for your applications",
      icon: <Key className="w-6 h-6" />,
      action: () => onNavigate("api-keys"),
      color: "bg-blue-500 hover:bg-blue-600",
      available: true
    },
    {
      title: "Add Model",
      description: "Configure new LLM providers and models",
      icon: <Plus className="w-6 h-6" />,
      action: () => onNavigate("models"),
      color: "bg-green-500 hover:bg-green-600",
      available: userRole === "Admin" || userRole === "Admin Viewer"
    },
    {
      title: "View Usage",
      description: "Monitor API usage and costs",
      icon: <BarChart3 className="w-6 h-6" />,
      action: () => onNavigate("usage"),
      color: "bg-purple-500 hover:bg-purple-600",
      available: true
    },
    {
      title: "Test Playground",
      description: "Interactive LLM testing environment",
      icon: <PlayCircle className="w-6 h-6" />,
      action: () => onNavigate("llm-playground"),
      color: "bg-orange-500 hover:bg-orange-600",
      available: true
    }
  ];

  const advancedFeatures = [
    {
      title: "Teams & Organizations",
      description: "Manage team access and billing",
      icon: <Users className="w-5 h-5" />,
      page: "teams"
    },
    {
      title: "Guardrails & Safety",
      description: "Configure content filtering",
      icon: <Shield className="w-5 h-5" />,
      page: "guardrails"
    },
    {
      title: "Caching & Performance",
      description: "Optimize response times",
      icon: <Zap className="w-5 h-5" />,
      page: "caching"
    },
    {
      title: "MCP Tools",
      description: "Model Context Protocol integration",
      icon: <Layers className="w-5 h-5" />,
      page: "mcp-servers"
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-sky-50 via-white to-indigo-50">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-r from-sky-600 via-blue-600 to-indigo-700 text-white">
        <div className="absolute inset-0 bg-black/10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center">
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="text-6xl animate-bounce">🏴‍☠️</div>
              <div>
                <h1 className="text-5xl md:text-7xl font-bold mb-4 bg-gradient-to-r from-white to-sky-200 bg-clip-text text-transparent">
                  Skypiea Gateway
                </h1>
                <p className="text-xl md:text-2xl text-sky-100 max-w-3xl mx-auto">
                  Modern LLM Proxy | Vision + Reasoning + Tools | 250+ Providers
                </p>
              </div>
            </div>

            <div className="flex flex-wrap justify-center gap-4 mb-12">
              <Badge variant="secondary" className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white border-yellow-300 px-4 py-2 text-sm shadow-lg animate-pulse">
                <Rocket className="w-4 h-4 mr-2 animate-bounce" />
                One Piece Powered
              </Badge>
              <Badge variant="secondary" className="bg-gradient-to-r from-blue-400 to-cyan-500 text-white border-blue-300 px-4 py-2 text-sm shadow-lg">
                <Eye className="w-4 h-4 mr-2" />
                Vision Ready
              </Badge>
              <Badge variant="secondary" className="bg-gradient-to-r from-green-400 to-emerald-500 text-white border-green-300 px-4 py-2 text-sm shadow-lg">
                <Activity className="w-4 h-4 mr-2" />
                Real-time Monitoring
              </Badge>
              <Badge variant="secondary" className="bg-gradient-to-r from-purple-500 to-pink-500 text-white border-purple-300 px-4 py-2 text-sm shadow-lg animate-pulse delay-300">
                <Globe className="w-4 h-4 mr-2 animate-spin" />
                250+ Providers
              </Badge>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                size="lg"
                className="bg-white text-sky-600 hover:bg-sky-50 font-semibold px-8 py-3 text-lg"
                onClick={() => onNavigate("api-keys")}
              >
                <Key className="w-5 h-5 mr-2" />
                Get Started
              </Button>
              <Button
                size="lg"
                variant="outline"
                className="border-white text-white hover:bg-white/10 font-semibold px-8 py-3 text-lg"
                onClick={() => onNavigate("llm-playground")}
              >
                <PlayCircle className="w-5 h-5 mr-2" />
                Try Playground
              </Button>
            </div>
          </div>
        </div>

        {/* One Piece themed floating elements */}
        <div className="absolute top-10 left-10 animate-bounce">
          <div className="w-20 h-20 bg-white/10 rounded-full flex items-center justify-center backdrop-blur-sm border border-white/20">
            <span className="text-3xl animate-spin">🏴‍☠️</span>
          </div>
        </div>
        <div className="absolute top-20 right-20 animate-pulse delay-500">
          <div className="w-16 h-16 bg-gradient-to-br from-yellow-400 to-orange-500 rounded-full flex items-center justify-center shadow-lg">
            <span className="text-2xl">⚡</span>
          </div>
        </div>
        <div className="absolute bottom-20 left-20 animate-pulse delay-1000">
          <div className="w-18 h-18 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
            <Eye className="w-8 h-8 text-white" />
          </div>
        </div>
        <div className="absolute bottom-10 right-10 animate-bounce delay-700">
          <div className="w-14 h-14 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg">
            <Globe className="w-6 h-6 text-white animate-spin" />
          </div>
        </div>

        {/* Animated waves at bottom */}
        <div className="absolute bottom-0 left-0 right-0">
          <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="w-full h-20">
            <path d="M0,60 C300,100 600,20 900,60 C1050,80 1200,40 1200,60 L1200,120 L0,120 Z" fill="rgba(255,255,255,0.1)" className="animate-pulse"></path>
            <path d="M0,80 C250,120 500,40 750,80 C900,100 1050,60 1200,80 L1200,120 L0,120 Z" fill="rgba(255,255,255,0.05)" className="animate-pulse delay-300"></path>
          </svg>
        </div>
      </div>

      {/* Metrics Dashboard */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <Card className="bg-gradient-to-br from-blue-500 to-blue-600 text-white border-0 shadow-lg hover:shadow-xl transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium opacity-90 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Total API Calls
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoading ? "..." : metrics.totalCalls.toLocaleString()}
              </div>
              <div className="flex items-center gap-1 mt-2">
                <TrendingUp className="w-3 h-3 text-green-300" />
                <span className="text-xs opacity-75">+23% from last month</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-500 to-green-600 text-white border-0 shadow-lg hover:shadow-xl transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium opacity-90 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Active Models
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoading ? "..." : metrics.activeModels}
              </div>
              <div className="flex items-center gap-1 mt-2">
                <TrendingUp className="w-3 h-3 text-green-300" />
                <span className="text-xs opacity-75">+5 new this week</span>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-purple-500 to-purple-600 text-white border-0 shadow-lg hover:shadow-xl transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium opacity-90 flex items-center gap-2">
                <Shield className="w-4 h-4" />
                System Uptime
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoading ? "..." : `${metrics.uptime}%`}
              </div>
              <Progress value={metrics.uptime} className="mt-2 bg-white/20" />
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-orange-500 to-orange-600 text-white border-0 shadow-lg hover:shadow-xl transition-shadow">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium opacity-90 flex items-center gap-2">
                <BarChart3 className="w-4 h-4" />
                Cost Savings
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">
                {isLoading ? "..." : `$${metrics.costSavings.toLocaleString()}`}
              </div>
              <div className="flex items-center gap-1 mt-2">
                <TrendingUp className="w-3 h-3 text-green-300" />
                <span className="text-xs opacity-75">12% reduction</span>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {quickActions.filter(action => action.available).map((action, index) => (
              <Card key={index} className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1 cursor-pointer border-2 hover:border-sky-200" onClick={action.action}>
                <CardHeader className="text-center pb-4">
                  <div className={`w-16 h-16 rounded-full ${action.color} flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform`}>
                    {action.icon}
                  </div>
                  <CardTitle className="text-lg">{action.title}</CardTitle>
                  <CardDescription className="text-sm">{action.description}</CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>

        {/* Advanced Features */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Advanced Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {advancedFeatures.map((feature, index) => (
              <Card key={index} className="hover:shadow-lg transition-shadow cursor-pointer border hover:border-indigo-200" onClick={() => onNavigate(feature.page)}>
                <CardHeader className="flex flex-row items-center space-y-0 pb-2">
                  <div className="p-2 bg-indigo-100 rounded-lg mr-3">
                    {feature.icon}
                  </div>
                  <div>
                    <CardTitle className="text-base">{feature.title}</CardTitle>
                    <CardDescription className="text-sm">{feature.description}</CardDescription>
                  </div>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>

        {/* System Status */}
        <Card className="bg-gradient-to-r from-gray-50 to-gray-100 border-2 border-gray-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-green-600" />
              System Status: All Systems Operational
            </CardTitle>
            <CardDescription>
              Skypiea Gateway is running smoothly with 99.8% uptime. All providers are responding normally.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">250+</div>
                <div className="text-sm text-gray-600">LLM Providers</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">&lt;100ms</div>
                <div className="text-sm text-gray-600">Avg Response Time</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">24/7</div>
                <div className="text-sm text-gray-600">Monitoring</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
