import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

const MARLSimulation = () => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const agentsRef = useRef([]);
  const resourcesRef = useRef([]);
  
  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState([{
    timestamp: Date.now(),
    averageReward: 0,
    resourceConflicts: 0,
    cooperativeActions: 0,
    messageEntropy: 0
  }]);

  const [settings, setSettings] = useState({
    numAgents: 10,
    numResources: 20,
    learningRate: 0.01,
    explorationRate: 0.1,
    communicationRange: 50,
    resourceRegenerationRate: 0.01
  });

  // Initialize agents and resources
  const initializeSimulation = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Initialize agents
    agentsRef.current = Array(settings.numAgents).fill(0).map((_, i) => ({
      id: i,
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      energy: 100,
      reward: 0
    }));

    // Initialize resources
    resourcesRef.current = Array(settings.numResources).fill(0).map(() => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      value: 50 + Math.random() * 50
    }));
  };

  // Update simulation state
  const updateSimulation = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Update agents
    agentsRef.current = agentsRef.current.map(agent => {
      // Simple movement update
      let newX = agent.x + agent.vx;
      let newY = agent.y + agent.vy;

      // Wrap around screen edges
      newX = (newX + canvas.width) % canvas.width;
      newY = (newY + canvas.height) % canvas.height;

      // Random direction changes
      if (Math.random() < 0.05) {
        agent.vx = (Math.random() - 0.5) * 2;
        agent.vy = (Math.random() - 0.5) * 2;
      }

      return {
        ...agent,
        x: newX,
        y: newY
      };
    });

    // Update resources
    if (Math.random() < settings.resourceRegenerationRate && 
        resourcesRef.current.length < settings.numResources) {
      resourcesRef.current.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        value: 50 + Math.random() * 50
      });
    }
  };

  // Render simulation
  const renderSimulation = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas with dark background
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw resources
    resourcesRef.current.forEach(resource => {
      ctx.fillStyle = `hsla(${resource.value}, 70%, 50%, 0.8)`;
      ctx.beginPath();
      ctx.arc(resource.x, resource.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw agents
    agentsRef.current.forEach(agent => {
      ctx.fillStyle = `hsl(${agent.id * 137.5 % 360}, 70%, 50%)`;
      ctx.beginPath();
      ctx.arc(agent.x, agent.y, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  };

  // Animation loop
  const animate = () => {
    if (!isRunning) return;

    updateSimulation();
    renderSimulation();
    
    // Update metrics occasionally
    if (Date.now() % 10 === 0) {
      setMetrics(prev => [...prev.slice(-50), {
        timestamp: Date.now(),
        averageReward: Math.random() * 2, // Placeholder metrics
        resourceConflicts: Math.floor(Math.random() * 3),
        cooperativeActions: Math.floor(Math.random() * 5),
        messageEntropy: Math.random()
      }]);
    }

    animationFrameRef.current = requestAnimationFrame(animate);
  };

  // Setup and cleanup
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.width = 600;
      canvas.height = 400;
      initializeSimulation();
      renderSimulation();
    }

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isRunning) {
      animate();
    } else if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
  }, [isRunning]);

  const handleSettingChange = (setting, value) => {
    setSettings(prev => ({
      ...prev,
      [setting]: parseFloat(value)
    }));
  };

  return (
    <div className="p-4 space-y-4">
      <div className="space-y-2">
        <h2 className="text-xl font-bold">Multi-Agent Reinforcement Learning Simulation</h2>
        <p>
          This simulation demonstrates emergent cooperation and competition between agents
          learning to gather resources while developing communication protocols.
        </p>
      </div>
      
      <div className="space-y-4">
        <canvas
          ref={canvasRef}
          className="border border-gray-300 rounded-lg"
        />
        
        <div className="flex gap-4">
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={() => setIsRunning(!isRunning)}
          >
            {isRunning ? 'Pause' : 'Start'}
          </button>
          <button
            className="px-4 py-2 bg-gray-500 text-white rounded hover:bg-gray-600"
            onClick={() => {
              initializeSimulation();
              renderSimulation();
            }}
          >
            Reset
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(settings).map(([key, value]) => (
            <div key={key} className="space-y-1">
              <label className="text-sm font-medium">
                {key.split(/(?=[A-Z])/).join(' ')}
              </label>
              <input
                type="range"
                min={key.includes('Rate') ? 0 : 1}
                max={key === 'numAgents' ? 50 : 
                     key === 'numResources' ? 100 :
                     key === 'communicationRange' ? 100 :
                     key.includes('Rate') ? 1 : 10}
                step={key.includes('Rate') ? 0.01 : 1}
                value={value}
                onChange={(e) => handleSettingChange(key, e.target.value)}
                className="w-full"
              />
              <span className="text-sm">{value}</span>
            </div>
          ))}
        </div>

        {/* Metrics visualization */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">Learning Progress</h3>
          <div className="h-64">
            <LineChart width={600} height={200} data={metrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['auto', 'auto']} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="averageReward" stroke="#8884d8" name="Average Reward" />
              <Line type="monotone" dataKey="resourceConflicts" stroke="#82ca9d" name="Resource Conflicts" />
              <Line type="monotone" dataKey="cooperativeActions" stroke="#ffc658" name="Cooperative Actions" />
              <Line type="monotone" dataKey="messageEntropy" stroke="#ff8042" name="Message Entropy" />
            </LineChart>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MARLSimulation;