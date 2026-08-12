import React, { useState, useEffect, useMemo } from 'react';
import {
  TrendingUp,
  TrendingDown,
  RotateCcw,
  Sparkles,
  Wifi,
  WifiOff,
  Activity,
  Info,
  Layers,
  HelpCircle
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// Default initial match state
const DEFAULT_INPUTS = {
  cum_runs: 120,
  cum_wickets: 3,
  balls_remaining: 30,
  runs_required: 45,
  required_run_rate: 9.0,
  current_run_rate: 8.0,
  recent_run_rate: 7.5,
  batting_team_prior: 0.5,
  bowling_team_prior: 0.5,
  event_type: 'other_runs',
};

export default function App() {
  const [inputs, setInputs] = useState(DEFAULT_INPUTS);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(null);
  const [autoSubmit, setAutoSubmit] = useState(true);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_BASE}/health_check`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'ok') {
            setBackendConnected(true);
            return;
          }
        }
        setBackendConnected(false);
      } catch (err) {
        setBackendConnected(false);
      }
    };
    checkConnection();
  }, []);

  useEffect(() => {
    if (backendConnected === true && history.length === 0) {
      triggerPrediction(DEFAULT_INPUTS, null);
    }
  }, [backendConnected]);

  const triggerPrediction = async (currentInputs, prevProba) => {
    setLoading(true);
    setError(null);

    const payload = {
      cum_runs: parseInt(currentInputs.cum_runs, 10),
      cum_wickets: parseInt(currentInputs.cum_wickets, 10),
      balls_remaining: parseInt(currentInputs.balls_remaining, 10),
      runs_required: parseInt(currentInputs.runs_required, 10),
      required_run_rate: parseFloat(currentInputs.required_run_rate),
      current_run_rate: parseFloat(currentInputs.current_run_rate),
      recent_run_rate: parseFloat(currentInputs.recent_run_rate),
      batting_team_prior: parseFloat(currentInputs.batting_team_prior),
      bowling_team_prior: parseFloat(currentInputs.bowling_team_prior),
      event_type: currentInputs.event_type,
      previous_proba: prevProba,
    };

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API Error: ${response.status} - ${errText}`);
      }

      const result = await response.json();

      const newRecord = {
        ballIndex: history.length + 1,
        ...payload,
        win_probability: result.win_probability,
        swing: result.swing !== undefined ? result.swing : null,
        explanation: result.explanation || null,
      };

      setHistory((prev) => [...prev, newRecord]);
      setInputs(prev => ({ ...prev, event_type: currentInputs.event_type }));
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to submit prediction request.');
    } finally {
      setLoading(false);
    }
  };

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    const lastProba = history.length > 0 ? history[history.length - 1].win_probability : null;
    await triggerPrediction(inputs, lastProba);
  };

  const handleInputChange = (field, value) => {
    setInputs((prev) => {
      const updated = { ...prev, [field]: value };
      if (field === 'cum_runs' || field === 'balls_remaining' || field === 'runs_required') {
        const cumRuns = parseInt(field === 'cum_runs' ? value : prev.cum_runs, 10) || 0;
        const ballsRemaining = parseInt(field === 'balls_remaining' ? value : prev.balls_remaining, 10) || 0;
        const runsRequired = parseInt(field === 'runs_required' ? value : prev.runs_required, 10) || 0;

        const ballsBowled = 120 - ballsRemaining;
        if (ballsBowled > 0) {
          updated.current_run_rate = parseFloat(((cumRuns / ballsBowled) * 6).toFixed(2));
        }
        if (ballsRemaining > 0) {
          updated.required_run_rate = parseFloat(((runsRequired / ballsRemaining) * 6).toFixed(2));
        }
      }
      return updated;
    });
  };

  const handlePresetClick = async (event) => {
    if (inputs.balls_remaining <= 0) return;

    const current = inputs;
    const isIllegalDelivery = event === 'wide' || event === 'noball';
    const newBallsRemaining = isIllegalDelivery
      ? current.balls_remaining
      : Math.max(0, current.balls_remaining - 1);
    const ballsBowled = 120 - newBallsRemaining;

    let runsAdded = 0;
    let wicketsAdded = 0;

    switch (event) {
      case 'dot_ball': runsAdded = 0; break;
      case 'single': runsAdded = 1; break;
      case 'two': runsAdded = 2; break;
      case 'three': runsAdded = 3; break;
      case 'four': runsAdded = 4; break;
      case 'six': runsAdded = 6; break;
      case 'wide': runsAdded = 1; break;
      case 'noball': runsAdded = 1; break;
      case 'wicket': wicketsAdded = 1; break;
      default: break;
    }

    const newCumRuns = current.cum_runs + runsAdded;
    const newCumWickets = Math.min(10, current.cum_wickets + wicketsAdded);
    const newRunsRequired = Math.max(0, current.runs_required - runsAdded);

    const newCurrentRunRate = ballsBowled > 0 ? parseFloat(((newCumRuns / ballsBowled) * 6).toFixed(2)) : 0;
    const newRequiredRunRate = newBallsRemaining > 0 ? parseFloat(((newRunsRequired / newBallsRemaining) * 6).toFixed(2)) : 0;

    const instantRR = runsAdded * 6;
    const newRecentRunRate = isIllegalDelivery
      ? current.recent_run_rate
      : parseFloat((current.recent_run_rate * 0.8 + instantRR * 0.2).toFixed(2));

    const apiEventType = ['single', 'two', 'three', 'wide', 'noball'].includes(event)
      ? 'other_runs'
      : event;

    const progressedInputs = {
      ...current,
      cum_runs: newCumRuns,
      cum_wickets: newCumWickets,
      balls_remaining: newBallsRemaining,
      runs_required: newRunsRequired,
      current_run_rate: newCurrentRunRate,
      required_run_rate: newRequiredRunRate,
      recent_run_rate: newRecentRunRate,
      event_type: apiEventType,
    };

    setInputs(progressedInputs);

    if (autoSubmit) {
      const lastProba = history.length > 0 ? history[history.length - 1].win_probability : null;
      await triggerPrediction(progressedInputs, lastProba);
    }
  };

  const handleReset = () => {
    setInputs(DEFAULT_INPUTS);
    setHistory([]);
    setError(null);
    if (backendConnected === true) {
      triggerPrediction(DEFAULT_INPUTS, null);
    }
  };

  const chartData = useMemo(() => {
    return history.map((item) => {
      const ballsBowled = 120 - item.balls_remaining;
      const overs = Math.floor(ballsBowled / 6);
      const ballInOver = ballsBowled % 6;
      return {
        label: `${overs}.${ballInOver}`,
        probability: Math.round(item.win_probability * 100),
        swing: item.swing ? parseFloat((item.swing * 100).toFixed(1)) : 0,
        runs: item.cum_runs,
        wickets: item.cum_wickets,
        event: item.event_type,
        explanation: item.explanation,
      };
    });
  }, [history]);

  const latestPrediction = history[history.length - 1];
  const currentProbability = latestPrediction ? latestPrediction.win_probability : null;
  const currentSwing = latestPrediction ? latestPrediction.swing : null;
  const currentExplanation = latestPrediction ? latestPrediction.explanation : null;

  const getPriorTier = (value) => {
    if (value < 0.4) return { label: 'Poor', color: 'text-brand-wicket' };
    if (value < 0.55) return { label: 'Average', color: 'text-brand-pine/70' };
    if (value < 0.7) return { label: 'Good', color: 'text-brand-moss' };
    return { label: 'Excellent', color: 'text-brand-amber' };
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-brand-card border-2 border-brand-pine/20 p-3 rounded-md shadow-xl text-xs max-w-xs font-sans">
          <p className="font-scoreboard font-semibold text-brand-pine/60 mb-1">Over {data.label}</p>
          <p className="text-brand-pine font-bold text-sm mb-1 font-scoreboard">
            Win Probability: {data.probability}%
          </p>
          <p className="text-brand-pine/80 font-medium">
            Score: {data.runs}/{data.wickets}
          </p>
          <p className="text-brand-pine/80">
            Event: <span className="capitalize font-semibold text-brand-amber">{data.event.replace('_', ' ')}</span>
          </p>
          {data.explanation && (
            <p className="text-brand-pine/70 mt-2 border-t border-brand-pine/10 pt-1.5 leading-relaxed italic text-[11px]">
              "{data.explanation}"
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  const formatOvers = (ballsRemaining) => {
    const bowled = 120 - ballsRemaining;
    return `${Math.floor(bowled / 6)}.${bowled % 6}`;
  };

  const presetButtonBase = "font-bold py-3 rounded text-xs transition duration-150 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer border-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine focus-visible:ring-offset-2 focus-visible:ring-offset-brand-parchment";

  return (
    <div className="min-h-screen bg-brand-parchment text-brand-pine flex flex-col font-sans">

      {/* Masthead — styled like the header of a cricket almanack, not a SaaS dashboard */}
      <header className="border-b-4 border-brand-pine px-6 py-5 flex flex-wrap justify-between items-end gap-4">
        <div className="flex items-baseline gap-3">
          <h1 className="text-3xl font-display font-black tracking-tight text-brand-pine">
            Crickcast
          </h1>
          <span className="text-xs font-scoreboard font-bold px-2 py-0.5 rounded bg-brand-yellow text-brand-pine-dark border border-brand-pine/20">
            T20
          </span>
        </div>
        <p className="text-xs text-brand-pine/60 font-medium italic font-display">
          A ball-by-ball win probability engine, with reasoning
        </p>

        <div className="flex items-center gap-3 ml-auto">
          <div className="flex items-center gap-2 bg-brand-card px-3 py-1.5 rounded-full border-2 border-brand-pine/15">
            {backendConnected === null ? (
              <>
                <div className="w-2 h-2 rounded-full bg-brand-amber animate-pulse" />
                <span className="text-xs text-brand-pine/60 font-medium">Connecting…</span>
              </>
            ) : backendConnected ? (
              <>
                <Wifi className="w-3.5 h-3.5 text-brand-moss" aria-hidden="true" />
                <span className="text-xs text-brand-moss font-semibold uppercase tracking-wide">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-brand-wicket" aria-hidden="true" />
                <span className="text-xs text-brand-wicket font-semibold uppercase tracking-wide">Offline</span>
                <button
                  onClick={() => window.location.reload()}
                  className="text-[10px] text-brand-pine/60 hover:text-brand-pine underline ml-1 cursor-pointer"
                >
                  Reconnect
                </button>
              </>
            )}
          </div>

          <button
            onClick={handleReset}
            aria-label="Reset the match simulation to its starting state"
            className="flex items-center gap-1.5 text-xs text-brand-pine hover:text-brand-pine-dark bg-brand-card hover:bg-brand-parchment-dim px-3 py-1.5 rounded-md border-2 border-brand-pine/15 hover:border-brand-pine/30 transition-all cursor-pointer font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine"
          >
            <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
            New Innings
          </button>
        </div>
      </header>

      <main className="flex-1 p-6 max-w-[1500px] w-full mx-auto flex flex-col gap-5">
        <p className="text-sm text-brand-pine/70 max-w-3xl font-medium">
          Tap what happens on the next ball — the scoreboard, win probability, and AI commentary
          update instantly, from a model trained on 3,000+ real T20 matches.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left: Ball Simulator */}
          <section className="lg:col-span-5 flex flex-col gap-5">
            <div className="bg-brand-card border-2 border-brand-pine/15 rounded-lg p-5 shadow-sm">
              <div className="flex items-center justify-between border-b-2 border-brand-pine/10 pb-3 mb-4">
                <h2 className="text-sm font-display font-bold uppercase tracking-wide text-brand-pine flex items-center gap-2">
                  <Activity className="w-4 h-4 text-brand-amber" aria-hidden="true" />
                  Ball-by-Ball Simulator
                </h2>
                <div className="flex items-center gap-2 text-xs">
                  <label className="text-brand-pine/70 cursor-pointer font-medium" htmlFor="auto-submit-toggle">Auto-predict</label>
                  <input
                    id="auto-submit-toggle"
                    type="checkbox"
                    checked={autoSubmit}
                    onChange={(e) => setAutoSubmit(e.target.checked)}
                    className="rounded border-brand-pine/30 text-brand-pine focus:ring-brand-pine w-4 h-4 cursor-pointer"
                  />
                </div>
              </div>

              <fieldset className="mb-5">
                <legend className="text-xs text-brand-pine/60 mb-2 font-semibold uppercase tracking-wide">Record next ball:</legend>
                <div className="grid grid-cols-5 gap-2 mb-2">
                  <button type="button" onClick={() => handlePresetClick('dot_ball')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine`}>
                    Dot
                  </button>
                  <button type="button" onClick={() => handlePresetClick('single')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine`}>
                    1 Run
                  </button>
                  <button type="button" onClick={() => handlePresetClick('two')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine`}>
                    2 Runs
                  </button>
                  <button type="button" onClick={() => handlePresetClick('three')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine`}>
                    3 Runs
                  </button>
                  <button type="button" onClick={() => handlePresetClick('wicket')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-wicket/10 hover:bg-brand-wicket/20 border-brand-wicket/40 text-brand-wicket`}>
                    Wicket
                  </button>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  <button type="button" onClick={() => handlePresetClick('four')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-moss/10 hover:bg-brand-moss/20 border-brand-moss/40 text-brand-moss`}>
                    Four
                  </button>
                  <button type="button" onClick={() => handlePresetClick('six')} disabled={loading || inputs.balls_remaining <= 0}
                    className={`${presetButtonBase} bg-brand-moss/15 hover:bg-brand-moss/25 border-brand-moss/50 text-brand-moss font-black`}>
                    Six
                  </button>
                  <button type="button" onClick={() => handlePresetClick('wide')} disabled={loading || inputs.balls_remaining <= 0}
                    title="Adds 1 run, does not consume a legal ball"
                    className={`${presetButtonBase} bg-brand-amber/10 hover:bg-brand-amber/20 border-brand-amber/40 text-brand-amber`}>
                    Wide
                  </button>
                  <button type="button" onClick={() => handlePresetClick('noball')} disabled={loading || inputs.balls_remaining <= 0}
                    title="Adds 1 run, does not consume a legal ball"
                    className={`${presetButtonBase} bg-brand-amber/10 hover:bg-brand-amber/20 border-brand-amber/40 text-brand-amber`}>
                    No Ball
                  </button>
                </div>
              </fieldset>

              <form onSubmit={handleManualSubmit} className="space-y-4">
                <details className="group">
                  <summary className="cursor-pointer select-none text-xs font-display font-bold text-brand-pine hover:text-brand-amber mb-3 flex items-center gap-2 list-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine rounded">
                    <span className="inline-block transition-transform duration-200 group-open:rotate-90 text-brand-amber">▶</span>
                    Scorer's Notebook — set a custom match state
                    <span className="text-brand-pine/50 font-normal font-sans hidden sm:inline">(optional)</span>
                  </summary>

                  <div className="notebook-lines rounded-md p-4 -mx-1 grid grid-cols-2 gap-4 bg-brand-parchment-dim/40 border border-brand-pine/10">
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Cumulative Runs</label>
                      <input type="number" min="0" max="400" value={inputs.cum_runs}
                        onChange={(e) => handleInputChange('cum_runs', parseInt(e.target.value, 10) || 0)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Wickets Down</label>
                      <input type="number" min="0" max="10" value={inputs.cum_wickets}
                        onChange={(e) => handleInputChange('cum_wickets', Math.min(10, parseInt(e.target.value, 10) || 0))}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Balls Remaining</label>
                      <input type="number" min="0" max="120" value={inputs.balls_remaining}
                        onChange={(e) => handleInputChange('balls_remaining', Math.min(120, parseInt(e.target.value, 10) || 0))}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Runs Required</label>
                      <input type="number" min="0" max="300" value={inputs.runs_required}
                        onChange={(e) => handleInputChange('runs_required', parseInt(e.target.value, 10) || 0)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="text-xs text-brand-pine/70 mb-1 flex items-center gap-1 font-medium">
                        Required Run Rate
                        <HelpCircle className="w-3 h-3 text-brand-pine/40 hover:text-brand-amber cursor-help" title="Runs per over needed to win from here" />
                      </label>
                      <input type="number" step="0.01" min="0" max="36" value={inputs.required_run_rate}
                        onChange={(e) => handleInputChange('required_run_rate', parseFloat(e.target.value) || 0)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="text-xs text-brand-pine/70 mb-1 flex items-center gap-1 font-medium">
                        Current Run Rate
                        <HelpCircle className="w-3 h-3 text-brand-pine/40 hover:text-brand-amber cursor-help" title="Runs per over scored so far" />
                      </label>
                      <input type="number" step="0.01" min="0" max="36" value={inputs.current_run_rate}
                        onChange={(e) => handleInputChange('current_run_rate', parseFloat(e.target.value) || 0)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Recent Run Rate (12 balls)</label>
                      <input type="number" step="0.01" min="0" max="36" value={inputs.recent_run_rate}
                        onChange={(e) => handleInputChange('recent_run_rate', parseFloat(e.target.value) || 0)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
                    </div>
                    <div>
                      <label htmlFor="event-type-select" className="block text-xs text-brand-pine/70 mb-1 font-medium">Event Context</label>
                      <select id="event-type-select" value={inputs.event_type}
                        onChange={(e) => handleInputChange('event_type', e.target.value)}
                        className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm focus:outline-none focus:border-brand-amber capitalize">
                        <option value="other_runs">Other Runs (1-3)</option>
                        <option value="dot_ball">Dot Ball</option>
                        <option value="four">Four</option>
                        <option value="six">Six</option>
                        <option value="wicket">Wicket</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-0.5 font-medium">Batting team's recent form</label>
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-[11px] font-semibold ${getPriorTier(inputs.batting_team_prior).color}`}>{getPriorTier(inputs.batting_team_prior).label}</span>
                        <span className="text-[10px] text-brand-pine/50 font-scoreboard">{(inputs.batting_team_prior * 100).toFixed(0)}%</span>
                      </div>
                      <input type="range" min="0" max="1" step="0.05" value={inputs.batting_team_prior}
                        onChange={(e) => handleInputChange('batting_team_prior', parseFloat(e.target.value))}
                        className="w-full accent-brand-pine h-1.5 rounded-lg cursor-pointer" />
                    </div>
                    <div>
                      <label className="block text-xs text-brand-pine/70 mb-0.5 font-medium">Bowling team's recent form</label>
                      <div className="flex items-center justify-between mb-1">
                        <span className={`text-[11px] font-semibold ${getPriorTier(inputs.bowling_team_prior).color}`}>{getPriorTier(inputs.bowling_team_prior).label}</span>
                        <span className="text-[10px] text-brand-pine/50 font-scoreboard">{(inputs.bowling_team_prior * 100).toFixed(0)}%</span>
                      </div>
                      <input type="range" min="0" max="1" step="0.05" value={inputs.bowling_team_prior}
                        onChange={(e) => handleInputChange('bowling_team_prior', parseFloat(e.target.value))}
                        className="w-full accent-brand-pine h-1.5 rounded-lg cursor-pointer" />
                    </div>
                  </div>
                </details>

                <button type="submit" disabled={loading || inputs.balls_remaining <= 0}
                  className="w-full bg-brand-pine hover:bg-brand-pine-dark text-brand-parchment font-display font-bold py-3 px-4 rounded-md text-sm transition-all shadow-md flex justify-center items-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber focus-visible:ring-offset-2 focus-visible:ring-offset-brand-parchment">
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-brand-parchment border-t-transparent rounded-full animate-spin" />
                      Predicting…
                    </>
                  ) : ('Predict Ball State')}
                </button>
              </form>
            </div>

            {error && (
              <div className="bg-brand-wicket/10 border-2 border-brand-wicket/30 text-brand-wicket p-4 rounded-lg text-xs leading-relaxed flex gap-2">
                <span className="text-sm" aria-hidden="true">⚠</span>
                <div>
                  <p className="font-semibold">Simulation error</p>
                  <p className="mt-0.5">{error}</p>
                </div>
              </div>
            )}
          </section>

          {/* Right: Scoreboard + Chart */}
          <section className="lg:col-span-7 flex flex-col gap-5">

            {/* The Scoreboard — signature element: an actual illustrated stadium scoreboard */}
            <div className="bg-brand-pine rounded-lg p-5 shadow-lg scoreboard-panel-glow border-4 border-brand-pine-dark">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-center pb-4 border-b-2 border-brand-parchment/10">
                {[
                  { label: 'Score', value: `${inputs.cum_runs}/${inputs.cum_wickets}` },
                  { label: 'Overs', value: formatOvers(inputs.balls_remaining) },
                  { label: 'Needed', value: inputs.runs_required },
                  { label: 'Req. R.R', value: inputs.required_run_rate.toFixed(1) },
                ].map((stat) => (
                  <div key={stat.label} className="bg-brand-pine-dark rounded p-2.5">
                    <p className="text-[9px] uppercase font-bold tracking-widest text-brand-parchment/50 mb-1">{stat.label}</p>
                    <p className="text-xl font-scoreboard font-bold text-brand-yellow scoreboard-glow">{stat.value}</p>
                  </div>
                ))}
              </div>

              <div className="flex flex-col md:flex-row justify-between items-center gap-6 py-6 border-b-2 border-brand-parchment/10" aria-live="polite" aria-atomic="true">
                <div className="flex-1 text-center md:text-left">
                  <p className="text-xs uppercase font-bold tracking-widest text-brand-parchment/50 mb-1">
                    Chasing Team Win Probability
                  </p>
                  <div className="inline-flex items-baseline gap-4 mt-1">
                    <span className="text-6xl md:text-7xl font-black font-scoreboard text-brand-yellow scoreboard-glow tracking-tight">
                      {currentProbability !== null ? `${(currentProbability * 100).toFixed(1)}%` : '—'}
                    </span>
                    {currentSwing !== null && (
                      <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border-2 ${currentSwing >= 0
                        ? 'bg-brand-moss/20 text-brand-moss border-brand-moss/40'
                        : 'bg-brand-wicket/20 text-brand-wicket border-brand-wicket/40'}`}>
                      {currentSwing >= 0 ? <TrendingUp className="w-3.5 h-3.5" aria-hidden="true" /> : <TrendingDown className="w-3.5 h-3.5" aria-hidden="true" />}
                      <span className="font-scoreboard">{currentSwing >= 0 ? '+' : '-'}{Math.abs(currentSwing * 100).toFixed(1)}%</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex gap-4 border-l-2 border-brand-parchment/10 pl-6 h-12 items-center text-xs hidden md:flex">
                  <div>
                    <p className="text-[9px] uppercase font-bold text-brand-parchment/40">Current R.R</p>
                    <p className="text-sm font-semibold font-scoreboard text-brand-parchment mt-0.5">{inputs.current_run_rate.toFixed(1)}</p>
                  </div>
                  <div className="h-4 border-r border-brand-parchment/10" />
                  <div>
                    <p className="text-[9px] uppercase font-bold text-brand-parchment/40">Recent R.R</p>
                    <p className="text-sm font-semibold font-scoreboard text-brand-parchment mt-0.5">{inputs.recent_run_rate.toFixed(1)}</p>
                  </div>
                </div>
              </div>

              <div className="mt-5 bg-brand-pine-dark rounded-md p-4" aria-live="polite">
                <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-brand-yellow">
                  <Sparkles className="w-4 h-4" aria-hidden="true" />
                  <span className="font-display">Commentary</span>
                </div>
                <p className="text-sm text-brand-parchment/90 leading-relaxed min-h-[40px]">
                  {currentExplanation
                    ? currentExplanation
                    : "Tap a ball event to generate live commentary explaining each swing in win probability."}
                </p>
              </div>
            </div>

            {/* Evolution chart */}
            <div className="bg-brand-card border-2 border-brand-pine/15 rounded-lg p-5 shadow-sm flex-1 flex flex-col min-h-[320px]">
              <h2 className="text-sm font-display font-bold uppercase tracking-wide text-brand-pine mb-4 flex items-center gap-2">
                <Layers className="w-4 h-4 text-brand-amber" aria-hidden="true" />
                Probability Through the Innings
              </h2>
              <div className="flex-1 w-full relative min-h-[240px]">
                {history.length <= 1 ? (
                  <div className="absolute inset-0 flex flex-col justify-center items-center text-center p-6 text-brand-pine/40 border-2 border-dashed border-brand-pine/15 rounded-lg">
                    <p className="text-sm font-medium">The curve builds as you play balls</p>
                    <p className="text-xs text-brand-pine/30 mt-1">Use the simulator to progress the innings over-by-over.</p>
                  </div>
                ) : null}
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(27,58,43,0.1)" />
                    <XAxis dataKey="label" stroke="rgba(27,58,43,0.4)" tick={{ fill: '#1b3a2b99', fontSize: 10 }} axisLine={{ stroke: 'rgba(27,58,43,0.2)' }} />
                    <YAxis domain={[0, 100]} stroke="rgba(27,58,43,0.4)" tick={{ fill: '#1b3a2b99', fontSize: 10 }} axisLine={{ stroke: 'rgba(27,58,43,0.2)' }} tickFormatter={(v) => `${v}%`} />
                    <Tooltip content={<CustomTooltip />} />
                    <ReferenceLine y={50} stroke="rgba(27,58,43,0.3)" strokeDasharray="4 4" label={{ value: '50/50', fill: '#1b3a2b66', fontSize: 10, position: 'top' }} />
                    <Line type="monotone" dataKey="probability" stroke="#1b3a2b" strokeWidth={3}
                      dot={{ r: 4, stroke: '#f2c230', strokeWidth: 2, fill: '#1b3a2b' }}
                      activeDot={{ r: 6, strokeWidth: 0, fill: '#f2c230' }} animationDuration={400} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Almanack Notes — model diagnostics, framed like reference pages rather than a dashboard */}
      <section className="bg-brand-pine-dark border-t-4 border-brand-pine py-8 mt-4">
        <div className="max-w-[1500px] w-full mx-auto px-6">
          <div className="flex items-center gap-2 mb-6 border-b-2 border-brand-parchment/10 pb-3">
            <Info className="w-5 h-5 text-brand-yellow" aria-hidden="true" />
            <h2 className="text-base font-display font-bold uppercase tracking-wide text-brand-parchment">
              Almanack Notes — Model Diagnostics
            </h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-brand-card border-2 border-brand-pine/10 rounded-lg p-5 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-display font-bold text-brand-pine mb-1">Probability Calibration Curve</h3>
                <p className="text-xs text-brand-pine/60 mb-4">Verifies whether the predicted win probabilities match actual historical match outcomes.</p>
                <div className="aspect-[4/3] bg-white border-2 border-brand-pine/10 rounded flex items-center justify-center overflow-hidden">
                  <img src="/calibration_curve.png" alt="Calibration curve showing predicted vs actual win probability"
                    className="max-h-full max-w-full object-contain p-1"
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span class="text-xs text-brand-pine/40">Calibration curve file missing</span>'; }} />
                </div>
              </div>
              <p className="text-xs text-brand-pine/80 mt-4 leading-relaxed bg-brand-parchment-dim/50 p-3 rounded border border-brand-pine/10">
                <span className="font-bold text-brand-amber">Reading it:</span> a perfectly calibrated model's line sits on the diagonal — this model's Expected Calibration Error dropped from 0.055 (baseline) to 0.009 (main model).
              </p>
            </div>

            <div className="bg-brand-card border-2 border-brand-pine/10 rounded-lg p-5 flex flex-col justify-between">
              <div>
                <h3 className="text-sm font-display font-bold text-brand-pine mb-1">Event-Impact Probability Swing</h3>
                <p className="text-xs text-brand-pine/60 mb-4">Average win-probability swing per ball, grouped by event type — measured directly from the model's own predictions.</p>
                <div className="aspect-[4/3] bg-white border-2 border-brand-pine/10 rounded flex items-center justify-center overflow-hidden">
                  <img src="/event_impact_swing.png" alt="Bar chart of average win-probability swing by event type"
                    className="max-h-full max-w-full object-contain p-1"
                    onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span class="text-xs text-brand-pine/40">Event impact file missing</span>'; }} />
                </div>
              </div>
              <p className="text-xs text-brand-pine/80 mt-4 leading-relaxed bg-brand-parchment-dim/50 p-3 rounded border border-brand-pine/10">
                <span className="font-bold text-brand-amber">Reading it:</span> wickets swing the probability hardest, sixes and fours swing it back — exactly what cricket sense would predict.
              </p>
            </div>
          </div>
        </div>
      </section>

      <footer className="border-t-2 border-brand-pine/15 py-4 bg-brand-parchment text-center text-xs text-brand-pine/50 font-medium">
        <p>Crickcast — Win Probability Engine, FastAPI + XGBoost.</p>
      </footer>
    </div>
  );
}