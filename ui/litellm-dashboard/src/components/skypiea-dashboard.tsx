"use client";

import React, { useState, useEffect } from "react";
import { Card, Button, Progress, Badge, Space, Statistic, Row, Col } from "antd";
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
              <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-white px-4 py-2 text-sm shadow-lg animate-pulse rounded-full flex items-center gap-2">
                <Rocket style={{ fontSize: 16 }} className="animate-bounce" />
                <span>One Piece Powered</span>
              </div>
              <div className="bg-gradient-to-r from-blue-400 to-cyan-500 text-white px-4 py-2 text-sm shadow-lg rounded-full flex items-center gap-2">
                <Eye style={{ fontSize: 16 }} />
                <span>Vision Ready</span>
              </div>
              <div className="bg-gradient-to-r from-green-400 to-emerald-500 text-white px-4 py-2 text-sm shadow-lg rounded-full flex items-center gap-2">
                <Activity style={{ fontSize: 16 }} />
                <span>Real-time Monitoring</span>
              </div>
              <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-2 text-sm shadow-lg animate-pulse delay-300 rounded-full flex items-center gap-2">
                <Globe style={{ fontSize: 16 }} className="animate-spin" />
                <span>250+ Providers</span>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                type="primary"
                size="large"
                style={{
                  background: 'white',
                  color: '#0284c7',
                  borderColor: 'white',
                  fontWeight: '600',
                  padding: '12px 32px',
                  fontSize: '18px'
                }}
                className="hover:bg-sky-50"
                onClick={() => onNavigate("api-keys")}
              >
                <Key style={{ fontSize: 20, marginRight: 8 }} />
                Get Started
              </Button>
              <Button
                size="large"
                style={{
                  borderColor: 'white',
                  color: 'white',
                  fontWeight: '600',
                  padding: '12px 32px',
                  fontSize: '18px'
                }}
                className="hover:bg-white/10"
                onClick={() => onNavigate("llm-playground")}
              >
                <PlayCircle style={{ fontSize: 20, marginRight: 8 }} />
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
          <div className="w-16 h-16 bg-gradient-to-br from-blue-400 to-cyan-500 rounded-full flex items-center justify-center shadow-lg">
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
        <Row gutter={[16, 16]} className="mb-12">
          <Col xs={24} sm={12} lg={6}>
            <Card
              style={{
                background: 'linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%)',
                border: 'none',
                color: 'white'
              }}
              className="shadow-lg"
            >
              <Statistic
                title={<Space><Activity style={{ fontSize: 16 }} />Total API Calls</Space>}
                value={isLoading ? "..." : metrics.totalCalls}
                valueStyle={{ color: 'white', fontSize: '24px' }}
                suffix={<Space className="text-green-300 text-xs"><TrendingUp />+23%</Space>}
              />
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card
              style={{
                background: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                border: 'none',
                color: 'white'
              }}
              className="shadow-lg"
            >
              <Statistic
                title={<Space><Zap style={{ fontSize: 16 }} />Active Models</Space>}
                value={isLoading ? "..." : metrics.activeModels}
                valueStyle={{ color: 'white', fontSize: '24px' }}
                suffix={<Space className="text-green-300 text-xs"><TrendingUp />+5</Space>}
              />
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card
              style={{
                background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
                border: 'none',
                color: 'white'
              }}
              className="shadow-lg"
            >
              <Statistic
                title={<Space><Shield style={{ fontSize: 16 }} />System Uptime</Space>}
                value={isLoading ? "..." : metrics.uptime}
                suffix="%"
                valueStyle={{ color: 'white', fontSize: '24px' }}
              />
              <Progress
                percent={metrics.uptime}
                showInfo={false}
                strokeColor="rgba(255,255,255,0.8)"
                trailColor="rgba(255,255,255,0.2)"
                className="mt-2"
              />
            </Card>
          </Col>

          <Col xs={24} sm={12} lg={6}>
            <Card
              style={{
                background: 'linear-gradient(135deg, #f97316 0%, #ea580c 100%)',
                border: 'none',
                color: 'white'
              }}
              className="shadow-lg"
            >
              <Statistic
                title={<Space><BarChart3 style={{ fontSize: 16 }} />Cost Savings</Space>}
                value={isLoading ? "..." : metrics.costSavings}
                prefix="$"
                valueStyle={{ color: 'white', fontSize: '24px' }}
                suffix={<Space className="text-green-300 text-xs"><TrendingUp />12%</Space>}
              />
            </Card>
          </Col>
        </Row>

        {/* Quick Actions */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Quick Actions</h2>
          <Row gutter={[16, 16]}>
            {quickActions.filter(action => action.available).map((action, index) => (
              <Col xs={24} sm={12} lg={6} key={index}>
                <Card
                  hoverable
                  className="text-center cursor-pointer transition-all duration-300 hover:shadow-lg"
                  onClick={action.action}
                  style={{ height: '100%' }}
                >
                  <div className="flex flex-col items-center p-4">
                    <div className={`w-16 h-16 rounded-full ${action.color} flex items-center justify-center mb-4 hover:scale-110 transition-transform`}>
                      {action.icon}
                    </div>
                    <h3 className="text-lg font-semibold mb-2">{action.title}</h3>
                    <p className="text-gray-600 text-sm">{action.description}</p>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </div>

        {/* Advanced Features */}
        <div className="mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">Advanced Features</h2>
          <Row gutter={[16, 16]}>
            {advancedFeatures.map((feature, index) => (
              <Col xs={24} sm={12} lg={6} key={index}>
                <Card
                  hoverable
                  className="cursor-pointer transition-shadow h-full"
                  onClick={() => onNavigate(feature.page)}
                  style={{ height: '100%' }}
                >
                  <div className="flex items-center p-4">
                    <div className="p-3 bg-indigo-100 rounded-lg mr-4">
                      {feature.icon}
                    </div>
                    <div>
                      <h4 className="text-base font-semibold mb-1">{feature.title}</h4>
                      <p className="text-gray-600 text-sm">{feature.description}</p>
                    </div>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </div>

        {/* System Status */}
        <Card
          style={{
            background: 'linear-gradient(to right, #f9fafb, #f3f4f6)',
            border: '2px solid #e5e7eb'
          }}
        >
          <div className="p-6">
            <div className="flex items-center gap-2 mb-4">
              <Activity style={{ fontSize: 20, color: '#16a34a' }} />
              <h3 className="text-lg font-semibold">System Status: All Systems Operational</h3>
            </div>
            <p className="text-gray-600 mb-6">
              Skypiea Gateway is running smoothly with 99.8% uptime. All providers are responding normally.
            </p>
            <Row gutter={[16, 16]}>
              <Col xs={24} md={8}>
                <div className="text-center">
                  <div className="text-3xl font-bold text-green-600">250+</div>
                  <div className="text-sm text-gray-600">LLM Providers</div>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-600">&lt;100ms</div>
                  <div className="text-sm text-gray-600">Avg Response Time</div>
                </div>
              </Col>
              <Col xs={24} md={8}>
                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-600">24/7</div>
                  <div className="text-sm text-gray-600">Monitoring</div>
                </div>
              </Col>
            </Row>
          </div>
        </Card>
      </div>
    </div>
  );
}
