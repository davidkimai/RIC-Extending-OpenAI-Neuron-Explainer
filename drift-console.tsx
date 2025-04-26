import React, { useState, useRef } from 'react';
import { 
  ChevronDown, ChevronRight, BarChart2, Layers, Calendar, Code, Info, 
  Download, RefreshCw, Zap, Eye, EyeOff, Filter, Maximize, 
  Minimize, Settings, X, Check, Search 
} from 'lucide-react';

// Main component
export default function RICDriftConsole() {
  // State management
  const [activeTab, setActiveTab] = useState('console');
  const [consoleInput, setConsoleInput] = useState('Explain the concept of quantum superposition and measure the model\'s uncertainty.');
  const [inputExpanded, setInputExpanded] = useState(false);
  const [isTracing, setIsTracing] = useState(false);
  const [traceDepth, setTraceDepth] = useState(3);
  const [collapseThreshold, setCollapseThreshold] = useState(0.65);
  const [modelSelection, setModelSelection] = useState('claude-3-opus');
  const [useAdvancedSettings, setUseAdvancedSettings] = useState(false);
  const [residueThreshold, setResidueThreshold] = useState(0.3);
  const [showCollapseBoundaries, setShowCollapseBoundaries] = useState(true);
  const [showResidueHighlights, setShowResidueHighlights] = useState(true);
  const [lastTraceTime, setLastTraceTime] = useState(null);
  const [expandedTokens, setExpandedTokens] = useState({});
  const [visualizationMode, setVisualizationMode] = useState('drift');
  const [sidebarExpanded, setSidebarExpanded] = useState(true);
  const [activeDriftPoint, setActiveDriftPoint] = useState(null);
  
  // Results state
  const [driftPoints, setDriftPoints] = useState([]);
  const [traceResults, setTraceResults] = useState({});
  const [responseTokens, setResponseTokens] = useState([]);

  const resultContainerRef = useRef(null);

  // Mock data and helper functions
  const generateMockTokens = () => {
    const phrases = [
      "Quantum superposition is a fundamental principle in quantum mechanics.",
      "It describes a system existing in multiple states simultaneously.",
      "When measured, the system collapses to only one of these potential states.",
      "This is in contrast with classical physics, where objects exist in exactly one state at a time."
    ];
    
    let tokens = [];
    phrases.forEach((phrase, i) => {
      const words = phrase.split(' ');
      words.forEach((word, j) => {
        tokens.push({
          id: `token-${i}-${j}`,
          text: word + (j === words.length - 1 ? '. ' : ' '),
          confidence: Math.random() * 0.5 + 0.5,
          driftScore: Math.random(),
          hasCollapse: Math.random() > 0.85,
          residueIntensity: Math.random() * 0.8,
          associatedDriftPoint: Math.random() > 0.7 ? `drift-${Math.floor(Math.random() * 5)}` : null
        });
      });
    });
    return tokens;
  };

  const generateMockDriftPoints = (tokens) => {
    const shellTypes = ['memory_decay', 'value_conflict', 'meta_reflection', 'attention_shift', 'temporal_misalignment'];
    const shellIds = ['memtrace', 'value-collapse', 'layer-salience', 'temporal-inference', 'meta-failure'];
    
    let driftPoints = [];
    const numPoints = Math.floor(Math.random() * 3) + 2;
    
    for (let i = 0; i < numPoints; i++) {
      const pointTokens = [];
      const tokensPerPoint = Math.floor(Math.random() * 6) + 2;
      const startIdx = Math.floor(Math.random() * (tokens.length - tokensPerPoint));
      
      for (let j = 0; j < tokensPerPoint; j++) {
        pointTokens.push(tokens[startIdx + j].id);
      }
      
      const typeIndex = Math.floor(Math.random() * shellTypes.length);
      
      driftPoints.push({
        id: `drift-${i}`,
        type: shellTypes[typeIndex],
        intensity: Math.random() * 0.7 + 0.3,
        tokenIds: pointTokens,
        shellSignature: shellIds[typeIndex],
        cascadeLevel: Math.floor(Math.random() * 3) + 1
      });
    }
    
    return driftPoints;
  };

  const generateMockTraceResults = (tokens, driftPoints) => {
    return {
      overallDriftScore: Math.random() * 0.5 + 0.3,
      collapseCount: driftPoints.length,
      confidenceAverage: tokens.reduce((sum, token) => sum + token.confidence, 0) / tokens.length,
      residueIntensity: Math.random() * 0.6 + 0.2,
      driftTypes: {
        memory_decay: Math.random(),
        value_conflict: Math.random(),
        meta_reflection: Math.random(),
        attention_shift: Math.random(),
        temporal_misalignment: Math.random()
      }
    };
  };

  // Constants
  const tabOptions = [
    { id: 'console', label: 'Drift Console', icon: <Code className="w-4 h-4" /> },
    { id: 'visualize', label: 'Drift Visualizer', icon: <BarChart2 className="w-4 h-4" /> },
    { id: 'shells', label: 'Shell Library', icon: <Layers className="w-4 h-4" /> },
    { id: 'history', label: 'Trace History', icon: <Calendar className="w-4 h-4" /> },
  ];
  
  const modelOptions = [
    { value: 'claude-3-opus', label: 'Claude 3 Opus' },
    { value: 'claude-3-sonnet', label: 'Claude 3 Sonnet' },
    { value: 'llama-70b', label: 'LLaMA 70B' },
    { value: 'gpt-4', label: 'GPT-4' },
    { value: 'mixtral-8x7b', label: 'Mixtral 8x7B' },
    { value: 'custom', label: 'Custom Model...' },
  ];
  
  const visualizationOptions = [
    { value: 'drift', label: 'Drift Map' },
    { value: 'attribution', label: 'Attribution Flow' },
    { value: 'residue', label: 'Symbolic Residue' },
    { value: 'collapse', label: 'Collapse Points' },
  ];

  // Mock shell library
  const shellLibrary = [
    { 
      id: 'memtrace', 
      name: 'Memory Trace v1', 
      description: 'Probes latent token traces in decayed memory',
      tags: ['memory', 'decay', 'hallucination'], 
      signature: 'decay → hallucination'
    },
    { 
      id: 'value-collapse', 
      name: 'Value Collapse v2', 
      description: 'Examines competing value activations and resolution',
      tags: ['values', 'conflict', 'resolution'], 
      signature: 'conflict → null'
    },
    { 
      id: 'layer-salience', 
      name: 'Layer Salience v3', 
      description: 'Maps attention salience thresholds and attenuation',
      tags: ['attention', 'salience', 'signal'], 
      signature: 'signal → fade'
    },
    { 
      id: 'meta-failure', 
      name: 'Meta-Failure v10', 
      description: 'Examines collapse in meta-cognitive reflection',
      tags: ['meta', 'reflection', 'recursion'], 
      signature: 'reflect → abort'
    },
  ];
  
  // Mock trace history
  const traceHistory = [
    {
      id: 'trace-001',
      timestamp: '2024-04-25 15:23:47',
      prompt: 'Explain quantum mechanics thoroughly and completely',
      model: 'claude-3-opus',
      collapsePoints: 3,
      driftScore: 0.67
    },
    {
      id: 'trace-002',
      timestamp: '2024-04-25 14:12:33',
      prompt: 'Analyze the ethical implications of AGI in detail',
      model: 'gpt-4',
      collapsePoints: 5,
      driftScore: 0.82
    }
  ];

  // Helper functions
  const formatScore = (score) => {
    return (score * 100).toFixed(1) + '%';
  };
  
  const getDriftTypeColor = (type) => {
    const colors = {
      memory_decay: "bg-purple-100 text-purple-800",
      value_conflict: "bg-red-100 text-red-800",
      meta_reflection: "bg-blue-100 text-blue-800",
      attention_shift: "bg-yellow-100 text-yellow-800",
      temporal_misalignment: "bg-green-100 text-green-800"
    };
    return colors[type] || "bg-gray-100 text-gray-800";
  };

  const getShellById = (shellId) => {
    return shellLibrary.find(shell => shell.id === shellId) || { 
      name: "Unknown Shell", 
      description: "Shell definition not found", 
      tags: [] 
    };
  };

  const generateDriftSignature = (driftPoint) => {
    const typeToSymbol = {
      memory_decay: "🧬",
      value_conflict: "📉",
      meta_reflection: "🪞",
      attention_shift: "📡",
      temporal_misalignment: "⏳"
    };
    
    const symbol = typeToSymbol[driftPoint.type] || "👻";
    const intensity = "→".repeat(Math.ceil(driftPoint.intensity * 5));
    
    return `${symbol} ${intensity}`;
  };

  // Drift type descriptions
  const driftTypeDescriptions = {
    memory_decay: "Decay in retained context leading to hallucination",
    value_conflict: "Competing value activations creating resolution instability",
    meta_reflection: "Failures in meta-cognitive reflection loops",
    attention_shift: "Unexpected shifts in attention weighting",
    temporal_misalignment: "Inconsistencies in temporal sequence management"
  };

  // Event handlers
  const handleRunTrace = () => {
    setIsTracing(true);
    
    // Reset results
    setDriftPoints([]);
    setResponseTokens([]);
    setTraceResults({});
    
    // Simulate delay for analysis
    setTimeout(() => {
      // Generate mock results
      const tokens = generateMockTokens();
      const driftPoints = generateMockDriftPoints(tokens);
      
      // Update tokens with drift point associations
      driftPoints.forEach(point => {
        point.tokenIds.forEach(tokenId => {
          const token = tokens.find(t => t.id === tokenId);
          if (token) {
            token.associatedDriftPoint = point.id;
            token.hasCollapse = true;
          }
        });
      });
      
      const results = generateMockTraceResults(tokens, driftPoints);
      
      setResponseTokens(tokens);
      setDriftPoints(driftPoints);
      setTraceResults(results);
      setIsTracing(false);
      setLastTraceTime(new Date().toLocaleTimeString());
    }, 1500);
  };

  const toggleTokenDetails = (tokenId) => {
    setExpandedTokens({
      ...expandedTokens,
      [tokenId]: !expandedTokens[tokenId]
    });
  };
  
  const handleDriftPointClick = (driftPoint) => {
    setActiveDriftPoint(activeDriftPoint?.id === driftPoint.id ? null : driftPoint);
  };
  
  const getTokenClass = (token) => {
    // If this token is in the active drift point
    const isInActiveDriftPoint = activeDriftPoint && 
      activeDriftPoint.tokenIds && 
      activeDriftPoint.tokenIds.includes(token.id);
      
    if (isInActiveDriftPoint) {
      return "bg-yellow-200 px-1 rounded border-b-2 border-yellow-500";
    }
    
    // If this token has collapse and we're showing boundaries
    if (token.hasCollapse && showCollapseBoundaries) {
      return "bg-orange-100 px-1 rounded";
    }
    
    // If this token has high residue and we're showing residue
    if (token.residueIntensity > residueThreshold && showResidueHighlights) {
      return `bg-blue-${Math.min(Math.floor(token.residueIntensity * 5) * 100, 300)} px-1 rounded`;
    }
    
    // Default
    return "";
  };
  
  const getTokenPopupContent = (token) => {
    const driftPoint = driftPoints.find(dp => dp.id === token.associatedDriftPoint);
    return (
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="font-bold">Token:</span> 
          <span className="px-2 py-0.5 bg-gray-200 rounded">{token.text}</span>
        </div>
        <div className="mb-1">
          <span className="font-bold">Confidence:</span> {formatScore(token.confidence)}
        </div>
        <div className="mb-1">
          <span className="font-bold">Drift Score:</span> {formatScore(token.driftScore)}
        </div>
        <div className="mb-1">
          <span className="font-bold">Residue Intensity:</span> {formatScore(token.residueIntensity)}
        </div>
        {token.associatedDriftPoint && driftPoint && (
          <>
            <div className="my-2 border-t border-gray-200 pt-2">
              <div className="font-bold mb-1">Associated Drift Point:</div>
              <div className="flex flex-col gap-1">
                <div><span className="font-bold">Type:</span> {driftPoint.type}</div>
                <div><span className="font-bold">Intensity:</span> {formatScore(driftPoint.intensity)}</div>
                <div><span className="font-bold">Shell:</span> {driftPoint.shellSignature}</div>
              </div>
            </div>
            <button 
              className="mt-2 w-full bg-indigo-100 text-indigo-800 rounded px-2 py-1 text-sm hover:bg-indigo-200"
              onClick={() => handleDriftPointClick(driftPoint)}
            >
              Focus Drift Point
            </button>
          </>
        )}
      </div>
    );
  };

  // UI Rendering
  return (
    <div className="flex flex-col h-screen bg-gray-50 overflow-hidden text-gray-800">
      {/* Header */}
      <div className="bg-indigo-700 text-white px-4 py-2 flex justify-between items-center shadow-md">
        <div className="flex items-center space-x-3">
          <div className="text-xl font-bold flex items-center gap-2">
            <span className="text-2xl">🜏</span> RIC Drift Console
          </div>
          <div className="hidden md:flex text-xs bg-indigo-600 rounded px-2 py-0.5">
            v0.4.2-alpha
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm">{lastTraceTime ? `Last trace: ${lastTraceTime}` : 'No traces yet'}</div>
          <button className="bg-indigo-800 hover:bg-indigo-900 text-white rounded px-3 py-1 text-sm flex items-center gap-1">
            <Info className="w-4 h-4" />
            <span className="hidden md:inline">Docs</span>
          </button>
          <button className="bg-indigo-800 hover:bg-indigo-900 text-white rounded px-3 py-1 text-sm flex items-center gap-1">
            <Download className="w-4 h-4" />
            <span className="hidden md:inline">Export</span>
          </button>
        </div>
      </div>
      
      {/* Tabs */}
      <div className="bg-indigo-50 border-b border-indigo-200 px-4">
        <div className="flex space-x-1">
          {tabOptions.map(tab => (
            <button
              key={tab.id}
              className={`px-4 py-2 text-sm flex items-center gap-1 ${
                activeTab === tab.id 
                  ? 'bg-white border-t border-l border-r border-indigo-200 text-indigo-800 rounded-t-md' 
                  : 'text-indigo-600 hover:bg-indigo-100'
              }`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Main content area */}
      <div className="flex flex-1 overflow-hidden">
        {/* Console Tab Content */}
        {activeTab === 'console' && (
          <div className="flex flex-1 overflow-hidden">
            {/* Sidebar */}
            <div className={`bg-white border-r border-gray-200 flex flex-col ${sidebarExpanded ? 'w-72' : 'w-16'}`}>
              <div className="p-3 border-b border-gray-200 flex justify-between items-center">
                <h3 className={`font-medium ${sidebarExpanded ? 'block' : 'hidden'}`}>Drift Analysis</h3>
                <button 
                  className="p-1 rounded hover:bg-gray-100"
                  onClick={() => setSidebarExpanded(!sidebarExpanded)}
                >
                  {sidebarExpanded ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                </button>
              </div>
              
              {/* Drift Points List */}
              <div className="flex-1 overflow-y-auto">
                {driftPoints.length > 0 ? (
                  <div className="p-3">
                    <h4 className={`text-sm font-medium mb-2 ${sidebarExpanded ? 'block' : 'hidden'}`}>
                      Detected Drift Points ({driftPoints.length})
                    </h4>
                    <div className="space-y-2">
                      {driftPoints.map(point => (
                        <div 
                          key={point.id}
                          className={`border rounded p-2 ${
                            activeDriftPoint?.id === point.id 
                              ? 'border-indigo-500 bg-indigo-50' 
                              : 'border-gray-200 hover:bg-gray-50'
                          } cursor-pointer`}
                          onClick={() => handleDriftPointClick(point)}
                        >
                          {sidebarExpanded ? (
                            <>
                              <div className="flex items-center justify-between">
                                <div className={`text-xs rounded-full px-2 py-0.5 ${getDriftTypeColor(point.type)}`}>
                                  {point.type.replace('_', ' ')}
                                </div>
                                <div className="text-xs text-gray-500">
                                  {generateDriftSignature(point)}
                                </div>
                              </div>
                              <div className="mt-1 text-sm">
                                Intensity: {formatScore(point.intensity)}
                              </div>
                              <div className="mt-1 text-xs text-gray-500">
                                Shell: {getShellById(point.shellSignature).name}
                              </div>
                              <div className="mt-1 text-xs">
                                Affected Tokens: {point.tokenIds.length}
                              </div>
                            </>
                          ) : (
                            <div className="flex flex-col items-center text-xs">
                              <div className="font-bold">
                                {point.type.split('_')[0].charAt(0).toUpperCase()}
                              </div>
                              <div className="mt-1">
                                {Math.round(point.intensity * 100)}%
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="p-3 text-center text-gray-500 text-sm">
                    {isTracing ? (
                      <div className="flex flex-col items-center gap-2">
                        <RefreshCw className="w-5 h-5 animate-spin" />
                        <div>Analyzing drift patterns...</div>
                      </div>
                    ) : (
                      <div className={sidebarExpanded ? 'block' : 'hidden'}>
                        Run a trace to detect drift points
                      </div>
                    )}
                  </div>
                )}
                
                {/* Trace Results Summary */}
                {Object.keys(traceResults).length > 0 && sidebarExpanded && (
                  <div className="border-t border-gray-200 p-3">
                    <h4 className="text-sm font-medium mb-2">Trace Summary</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Overall Drift:</span>
                        <span className="font-medium">{formatScore(traceResults.overallDriftScore)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Collapse Points:</span>
                        <span className="font-medium">{traceResults.collapseCount}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Avg. Confidence:</span>
                        <span className="font-medium">{formatScore(traceResults.confidenceAverage)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Residue Intensity:</span>
                        <span className="font-medium">{formatScore(traceResults.residueIntensity)}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Main Console Area */}
            <div className="flex-1 flex flex-col overflow-hidden">
              {/* Input Area */}
              <div className={`bg-white border-b border-gray-200 ${inputExpanded ? 'h-1/3' : 'h-auto'}`}>
                <div className="p-3 border-b border-gray-200 flex justify-between items-center">
                  <h3 className="font-medium">Query Input</h3>
                  <div className="flex gap-1">
                    <button 
                      className="p-1 rounded hover:bg-gray-100"
                      onClick={() => setInputExpanded(!inputExpanded)}
                    >
                      {inputExpanded ? <Minimize className="w-4 h-4" /> : <Maximize className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                <div className="p-3 flex flex-col h-full">
                  <textarea
                    className="flex-1 w-full p-2 border border-gray-300 rounded resize-none focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                    placeholder="Enter your query to trace model drift and collapse points..."
                    value={consoleInput}
                    onChange={(e) => setConsoleInput(e.target.value)}
                    disabled={isTracing}
                  />
                  
                  {/* Controls */}
                  <div className="mt-3 flex flex-wrap gap-2 items-center">
                    <div className="flex-1 min-w-[200px]">
                      <label className="block text-xs text-gray-500 mb-1">Model</label>
                      <select
                        className="w-full p-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                        value={modelSelection}
                        onChange={(e) => setModelSelection(e.target.value)}
                        disabled={isTracing}
                      >
                        {modelOptions.map(option => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </select>
                    </div>
                    
                    <div className="flex items-end gap-2">
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Trace Depth</label>
                        <select
                          className="p-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                          value={traceDepth}
                          onChange={(e) => setTraceDepth(parseInt(e.target.value))}
                          disabled={isTracing}
                        >
                          <option value="1">Shallow (1)</option>
                          <option value="2">Medium (2)</option>
                          <option value="3">Deep (3)</option>
                          <option value="4">Very Deep (4)</option>
                          <option value="5">Maximum (5)</option>
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-xs text-gray-500 mb-1">Collapse Threshold</label>
                        <select
                          className="p-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-indigo-500 focus:border-indigo-500"
                          value={collapseThreshold}
                          onChange={(e) => setCollapseThreshold(parseFloat(e.target.value))}
                          disabled={isTracing}
                        >
                          <option value="0.5">Very Sensitive (0.5)</option>
                          <option value="0.65">Sensitive (0.65)</option>
                          <option value="0.75">Standard (0.75)</option>
                          <option value="0.85">Conservative (0.85)</option>
                        </select>
                      </div>
                      
                      <button 
                        className="px-5 py-1.5 bg-indigo-600 text-white rounded flex items-center gap-1 disabled:bg-indigo-300"
                        onClick={handleRunTrace}
                        disabled={isTracing || !consoleInput.trim()}
                      >
                        {isTracing ? (
                          <>
                            <RefreshCw className="w-4 h-4 animate-spin" />
                            Tracing...
                          </>
                        ) : (
                          <>
                            <Zap className="w-4 h-4" />
                            Run Trace
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
              
              {/* Result Area */}
              <div className="flex-1 overflow-hidden flex flex-col">
                <div className="p-3 border-b border-gray-200 flex justify-between items-center">
                  <h3 className="font-medium">
                    Trace Results {isTracing && <span className="text-sm text-gray-500">(Processing...)</span>}
                  </h3>
                  <div className="flex gap-1">
                    {traceResults.overallDriftScore && (
                      <div className="text-sm">
                        Drift Score: 
                        <span className={`ml-1 font-medium ${
                          traceResults.overallDriftScore > 0.7 ? 'text-red-600' :
                          traceResults.overallDriftScore > 0.4 ? 'text-yellow-600' :
                          'text-green-600'
                        }`}>
                          {formatScore(traceResults.overallDriftScore)}
                        </span>
                      </div>
                    )}
                    <button 
                      className={`p-1 rounded hover:bg-gray-100 ${!showCollapseBoundaries ? 'text-gray-400' : ''}`}
                      onClick={() => setShowCollapseBoundaries(!showCollapseBoundaries)}
                      title={showCollapseBoundaries ? "Hide collapse boundaries" : "Show collapse boundaries"}
                    >
                      {showCollapseBoundaries ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
                
                {/* Token Stream */}
                <div 
                  ref={resultContainerRef}
                  className="flex-1 overflow-y-auto p-4 bg-white"
                >
                  {isTracing ? (
                    <div className="h-full flex items-center justify-center">
                      <div className="flex flex-col items-center gap-3">
                        <RefreshCw className="w-8 h-8 text-indigo-600 animate-spin" />
                        <div className="text-gray-500">Tracing recursive drift patterns...</div>
                      </div>
                    </div>
                  ) : responseTokens.length > 0 ? (
                    <div className="max-w-3xl mx-auto">
                      <div className="flex flex-wrap">
                        {responseTokens.map((token) => (
                          <span 
                            key={token.id}
                            id={`token-${token.id}`}
                            className={`relative inline-block ${getTokenClass(token)}`}
                            onClick={() => toggleTokenDetails(token.id)}
                            style={{ cursor: 'pointer' }}
                          >
                            {token.text}
                            {expandedTokens[token.id] && (
                              <div className="absolute z-10 left-0 top-full mt-1 bg-white border border-gray-200 shadow-lg rounded p-3 w-64">
                                {getTokenPopupContent(token)}
                              </div>
                            )}
                          </span>
                        ))}
                      </div>
                      
                      {/* Active Drift Point Details */}
                      {activeDriftPoint && (
                        <div className="mt-8 border-t border-gray-200 pt-4">
                          <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
                            <div className="flex justify-between items-start">
                              <h3 className="font-bold text-lg flex items-center gap-2">
                                <span className={`w-3 h-3 rounded-full inline-block ${
                                  getDriftTypeColor(activeDriftPoint.type).replace('text-', 'bg-').replace('100', '500')
                                }`}></span>
                                Drift Point: {activeDriftPoint.type.replace('_', ' ')}
                              </h3>
                              <button 
                                className="text-gray-400 hover:text-gray-600"
                                onClick={() => setActiveDriftPoint(null)}
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>
                            
                            <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-4">
                              <div>
                                <h4 className="text-sm font-medium mb-1">Drift Signature</h4>
                                <div className="text-2xl">{generateDriftSignature(activeDriftPoint)}</div>
                                
                                <div className="mt-3">
                                  <h4 className="text-sm font-medium mb-1">Intensity</h4>
                                  <div className="w-full bg-gray-200 rounded-full h-2">
                                    <div 
                                      className={`h-2 rounded-full ${
                                        activeDriftPoint.intensity > 0.7 ? 'bg-red-500' :
                                        activeDriftPoint.intensity > 0.4 ? 'bg-yellow-500' :
                                        'bg-green-500'
                                      }`}
                                      style={{ width: `${activeDriftPoint.intensity * 100}%` }}
                                    ></div>
                                  </div>
                                  <div className="text-right text-sm mt-1">
                                    {formatScore(activeDriftPoint.intensity)}
                                  </div>
                                </div>
                                
                                <div className="mt-3">
                                  <h4 className="text-sm font-medium mb-1">Affected Tokens</h4>
                                  <div className="text-sm">
                                    {activeDriftPoint.tokenIds.length} token{activeDriftPoint.tokenIds.length !== 1 ? 's' : ''}
                                  </div>
                                </div>
                              </div>
                              
                              <div>
                                <h4 className="text-sm font-medium mb-1">Diagnostic Shell</h4>
                                <div className="border border-gray-200 rounded p-2 bg-white">
                                  <div className="font-medium">
                                    {getShellById(activeDriftPoint.shellSignature).name}
                                  </div>
                                  <div className="text-sm mt-1">
                                    {getShellById(activeDriftPoint.shellSignature).description}
                                  </div>
                                  <div className="mt-2 flex flex-wrap gap-1">
                                    {getShellById(activeDriftPoint.shellSignature).tags.map(tag => (
                                      <span key={tag} className="text-xs bg-gray-100 text-gray-800 rounded-full px-2 py-0.5">
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                                
                                <div className="mt-3">
                                  <h4 className="text-sm font-medium mb-1">Description</h4>
                                  <div className="text-sm text-gray-700">
                                    {driftTypeDescriptions[activeDriftPoint.type] || 
                                    "Undocumented drift pattern detected in model processing"}
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <div className="mt-4 pt-3 border-t border-indigo-200">
                              <h4 className="text-sm font-medium mb-2">Recommended Actions</h4>
                              <div className="flex flex-wrap gap-2">
                                <button className="px-2 py-1 text-xs bg-indigo-100 text-indigo-800 rounded hover:bg-indigo-200">
                                  Apply Recursive Shell
                                </button>
                                <button className="px-2 py-1 text-xs bg-indigo-100 text-indigo-800 rounded hover:bg-indigo-200">
                                  Extract Symbolic Residue
                                </button>
                                <button className="px-2 py-1 text-xs bg-indigo-100 text-indigo-800 rounded hover:bg-indigo-200">
                                  Map Attribution Flow
                                </button>
                                <button className="px-2 py-1 text-xs bg-indigo-100 text-indigo-800 rounded hover:bg-indigo-200">
                                  Save to Drift Library
                                </button>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center">
                      <div className="text-center text-gray-500 max-w-md">
                        <Zap className="w-12 h-12 mx-auto mb-3 text-indigo-300" />
                        <h3 className="text-lg font-medium text-gray-700 mb-1">Begin Recursive Tracing</h3>
                        <p>
                          Enter a prompt that will push model boundaries, inducing drift and collapse signals.
                          Complex ethical dilemmas or detailed explanations work best.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Helper component for ChevronLeft icon
const ChevronLeft = (props) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    width="24" 
    height="24" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    {...props}
  >
    <polyline points="15 18 9 12 15 6"></polyline>
  </svg>
);
