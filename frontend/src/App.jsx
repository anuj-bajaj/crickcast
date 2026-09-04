import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  RotateCcw,
  Undo2,
  Sparkles,
  Wifi,
  WifiOff,
  Activity,
  Info,
  Layers,
  HelpCircle,
  ArrowRight,
  Radio,
  Settings2,
  Flag,
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { motion, AnimatePresence, useInView, useAnimationControls } from 'framer-motion';

// In dev this is unset, so requests stay relative ('/predict') and go
// through the Vite dev-server proxy (see vite.config.js). In production
// Vite bakes VITE_API_BASE_URL in at build time, so requests go straight
// to the deployed FastAPI backend instead of the static frontend host,
// which doesn't have a /predict route.
const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

// Session-only on purpose: a chase you're mid-way through simulating
// shouldn't vanish on an accidental refresh, but it also shouldn't still be
// sitting there weeks later for someone opening the site fresh — sessionStorage
// clears itself when the tab closes, which matches how long this state
// should actually live.
const STORAGE_KEY = 'crickcast_session_v1';

// Module-level (not per-render) so it's a stable reference for the
// scorecardChips useMemo below — a `const` re-created inside the component
// body on every render would defeat that memoization if added as a
// dependency, which is why it lives here instead.
const LEGAL_BALL_EVENTS = new Set(['dot_ball', 'single', 'two', 'three', 'four', 'six', 'wicket']);

// Nothing is pre-filled here on purpose — the whole point of the setup
// stage is that the user supplies the target and (optionally) the current
// match state, and every derived number is computed from that.
const EMPTY_INPUTS = {
  target: '',
  cum_runs: 0,
  cum_wickets: 0,
  overs_completed: 0,
  balls_this_over: 0,
  recent_run_rate: null, // no ball history yet to derive this from
  batting_team_prior: 0.5,
  bowling_team_prior: 0.5,
  event_type: 'other_runs',
  // '' means "custom" — sliders below are hand-set, no real team behind
  // them, and no team name gets sent to the explanation prompt.
  batting_team_name: '',
  bowling_team_name: '',
};

// Single source of truth for every "computed" number — balls remaining,
// runs required, current/required run rate. The user only ever edits the
// raw inputs (target, score, wickets, overs); everything else is derived
// fresh from those on every render instead of being separately editable
// state that can drift out of sync.
function deriveMatchStats(inputs) {
  const target = inputs.target === '' || inputs.target === null || inputs.target === undefined
    ? null
    : parseInt(inputs.target, 10);
  const cumRuns = parseInt(inputs.cum_runs, 10) || 0;
  const cumWickets = Math.min(10, parseInt(inputs.cum_wickets, 10) || 0);
  const oversCompleted = Math.max(0, parseInt(inputs.overs_completed, 10) || 0);
  const ballsThisOver = Math.min(5, Math.max(0, parseInt(inputs.balls_this_over, 10) || 0));
  const ballsBowled = Math.min(120, oversCompleted * 6 + ballsThisOver);
  const ballsRemaining = Math.max(0, 120 - ballsBowled);
  const runsRequired = target !== null ? Math.max(0, target - cumRuns) : null;
  const currentRunRate = ballsBowled > 0 ? (cumRuns / ballsBowled) * 6 : 0;
  const requiredRunRate = target === null
    ? null
    : (ballsRemaining > 0 ? (runsRequired / ballsRemaining) * 6 : 99);

  return { target, cumRuns, cumWickets, oversCompleted, ballsThisOver, ballsBowled, ballsRemaining, runsRequired, currentRunRate, requiredRunRate };
}

// A real hover/focus tooltip, styled to match the brand — the native
// `title` attribute on an SVG icon is unreliable across browsers (some
// require a <title> child element rather than the attribute, and even
// where it works the delay/styling is generic OS chrome, easy to miss on
// a 12px icon). This is small, keyboard-accessible (shows on focus too,
// not just hover), and actually visible.
function InfoTooltip({ text }) {
  const [show, setShow] = useState(false);
  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      <HelpCircle
        tabIndex={0}
        aria-label={text}
        className="w-3.5 h-3.5 text-brand-pine/40 hover:text-brand-amber focus-visible:text-brand-amber cursor-help focus-visible:outline-none transition-colors"
      />
      <AnimatePresence>
        {show && (
          <motion.span
            role="tooltip"
            initial={{ opacity: 0, y: 4, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 4, scale: 0.96 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 bg-brand-pine text-brand-parchment text-[11px] leading-snug font-sans font-normal normal-case tracking-normal rounded-md px-2.5 py-1.5 shadow-lg pointer-events-none"
          >
            {text}
            <span className="absolute top-full left-1/2 -translate-x-1/2 -mt-px w-0 h-0 border-4 border-transparent border-t-brand-pine" aria-hidden="true" />
          </motion.span>
        )}
      </AnimatePresence>
    </span>
  );
}

// Small inline logo mark — a cricket ball with a stitched seam, no external asset needed
function LogoMark({ size = 36 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <circle cx="20" cy="20" r="18" fill="var(--color-brand-pine)" stroke="var(--color-brand-yellow)" strokeWidth="2" />
      <path d="M11 12 Q20 20 11 28" stroke="var(--color-brand-parchment)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M29 12 Q20 20 29 28" stroke="var(--color-brand-parchment)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
      <path d="M11 12 Q20 20 11 28" stroke="var(--color-brand-parchment)" strokeWidth="0.5" fill="none" strokeDasharray="1,1.5" />
      <path d="M29 12 Q20 20 29 28" stroke="var(--color-brand-parchment)" strokeWidth="0.5" fill="none" strokeDasharray="1,1.5" />
    </svg>
  );
}

// One-shot confetti burst for a won chase — plain framer-motion, no
// external library. Mounted once (the parent only renders this when
// chaseWon first becomes true, and a chase can only be won once per
// innings), so it doesn't need its own start/stop control — each piece
// just plays its fall-and-fade once and settles off-screen.
function Confetti() {
  const pieces = useMemo(() => Array.from({ length: 28 }, (_, i) => ({
    id: i,
    left: Math.random() * 100,
    delay: Math.random() * 0.35,
    duration: 1.8 + Math.random() * 1.3,
    rotate: (Math.random() - 0.5) * 720,
    color: ['bg-brand-yellow', 'bg-brand-amber', 'bg-brand-moss', 'bg-brand-parchment'][i % 4],
    width: 5 + Math.random() * 5,
  })), []);
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-lg" aria-hidden="true">
      {pieces.map((p) => (
        <motion.span
          key={p.id}
          initial={{ y: -20, opacity: 1, rotate: 0 }}
          animate={{ y: 420, opacity: [1, 1, 0], rotate: p.rotate }}
          transition={{ duration: p.duration, delay: p.delay, ease: 'easeIn' }}
          className={`absolute rounded-sm ${p.color}`}
          style={{ left: `${p.left}%`, width: p.width, height: p.width * 1.7 }}
        />
      ))}
    </div>
  );
}

// Animates a number counting up/down to its target value whenever it changes
function useAnimatedNumber(target, duration = 600) {
  const [display, setDisplay] = useState(target);
  const rafRef = useRef(null);
  const startRef = useRef(null);
  const fromRef = useRef(target);

  useEffect(() => {
    if (target === null || target === undefined) return;
    fromRef.current = display;
    startRef.current = null;

    const step = (timestamp) => {
      if (!startRef.current) startRef.current = timestamp;
      const progress = Math.min((timestamp - startRef.current) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(fromRef.current + (target - fromRef.current) * eased);
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(step);
      }
    };

    rafRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target]);

  return display;
}

// Counts up to `value` once it scrolls into view — used for the Almanack stat tiles
function AnimatedStat({ value, decimals = 0, suffix = '' }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const [display, setDisplay] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let raf;
    let start = null;
    const duration = 1100;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(value * eased);
      if (progress < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [inView, value]);

  return <span ref={ref}>{display.toFixed(decimals)}{suffix}</span>;
}

// Reusable "Scorer's Notebook" field set — used both to seed a starting state
// (setup stage) and to alter the state mid-chase (live stage).
//
// Only raw, scoreboard-legible facts are editable here (runs, wickets,
// overs). Balls remaining / runs required / current & required run rate are
// NOT inputs — they're derived (see deriveMatchStats) and shown read-only
// so they can never drift out of sync with what the user actually typed.
//
// `showEventType` is only turned on for the live "Alter state" editor: mid-
// chase there IS a previous prediction to swing away from, so an event
// context is meaningful there. Pre-first-ball (setup), there's nothing to
// swing from yet, so it's omitted entirely.
function ScorersNotebookFields({ inputs, onChange, showEventType = false }) {
  return (
    <div className="notebook-lines rounded-md p-4 -mx-1 grid grid-cols-2 md:grid-cols-3 gap-4 bg-brand-parchment-dim/40 border border-brand-pine/10">
      <div>
        <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Cumulative Runs</label>
        <input type="number" min="0" max="400" value={inputs.cum_runs} onChange={(e) => onChange('cum_runs', e.target.value)} className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
      </div>
      <div>
        <label className="block text-xs text-brand-pine/70 mb-1 font-medium">Wickets Down</label>
        <input type="number" min="0" max="10" value={inputs.cum_wickets} onChange={(e) => onChange('cum_wickets', e.target.value)} className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
      </div>
      <div>
        <label className="text-xs text-brand-pine/70 mb-1 flex items-center gap-1 font-medium">Overs Completed <InfoTooltip text="Full overs bowled so far this innings" /></label>
        <input type="number" min="0" max="19" value={inputs.overs_completed} onChange={(e) => onChange('overs_completed', e.target.value)} className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
      </div>
      <div>
        <label className="text-xs text-brand-pine/70 mb-1 flex items-center gap-1 font-medium">Balls (this over) <InfoTooltip text="Legal deliveries bowled in the current, unfinished over" /></label>
        <input type="number" min="0" max="5" value={inputs.balls_this_over} onChange={(e) => onChange('balls_this_over', e.target.value)} className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm font-scoreboard focus:outline-none focus:border-brand-amber" />
      </div>
      {showEventType && (
        <div>
          <label htmlFor="event-type-select" className="block text-xs text-brand-pine/70 mb-1 font-medium">Event Context</label>
          <select id="event-type-select" value={inputs.event_type} onChange={(e) => onChange('event_type', e.target.value)} className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm focus:outline-none focus:border-brand-amber capitalize">
            <option value="other_runs">Other Runs (1-3)</option>
            <option value="dot_ball">Dot Ball</option>
            <option value="four">Four</option>
            <option value="six">Six</option>
            <option value="wicket">Wicket</option>
          </select>
        </div>
      )}
      <div>
        <label className="block text-xs text-brand-pine/70 mb-0.5 font-medium">Batting team's recent form</label>
        <div className="flex items-center justify-between mb-1"><span className="text-[11px] font-semibold text-brand-pine/70">{(inputs.batting_team_prior * 100).toFixed(0)}%</span></div>
        <input type="range" min="0" max="1" step="0.05" value={inputs.batting_team_prior} onChange={(e) => onChange('batting_team_prior', parseFloat(e.target.value))} className="w-full accent-brand-pine h-1.5 rounded-lg cursor-pointer" />
      </div>
      <div>
        <label className="block text-xs text-brand-pine/70 mb-0.5 font-medium">Bowling team's recent form</label>
        <div className="flex items-center justify-between mb-1"><span className="text-[11px] font-semibold text-brand-pine/70">{(inputs.bowling_team_prior * 100).toFixed(0)}%</span></div>
        <input type="range" min="0" max="1" step="0.05" value={inputs.bowling_team_prior} onChange={(e) => onChange('bowling_team_prior', parseFloat(e.target.value))} className="w-full accent-brand-pine h-1.5 rounded-lg cursor-pointer" />
      </div>
    </div>
  );
}

export default function App() {
  const [matchStarted, setMatchStarted] = useState(false);
  const [inputs, setInputs] = useState(EMPTY_INPUTS);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [backendConnected, setBackendConnected] = useState(null);
  const [autoSubmit, setAutoSubmit] = useState(true);
  const [editingState, setEditingState] = useState(false);
  // Snapshot of `inputs` taken the instant the Alter State editor is
  // opened — NOT re-derived from `inputs` at submit time, because
  // ScorersNotebookFields writes every keystroke straight into `inputs`
  // (the same state object used as "current"). Without a separately
  // captured snapshot, by the time handleManualSubmit runs, the "before"
  // state has already been overwritten by the user's edits, so before ===
  // after on every field — which is exactly how an Alter State jump of
  // e.g. 30 runs and 3 overs got sent to the explanation layer as "0 legal
  // deliveries, wide/no-ball, every number unchanged" instead of the real,
  // possibly large, before/after delta it actually represents.
  const preEditInputsRef = useRef(null);
  // Monotonically increasing id for each history entry — see
  // triggerPrediction's use of it. Needs to survive across renders as a
  // ref (not state) since it's read-then-incremented synchronously and
  // must never trigger its own re-render.
  const recordIdCounter = useRef(0);
  // Controls the Scorer's Notebook <details> on the setup page — normally
  // left to the browser's native toggle, but a Quick Start chip that fills
  // in non-zero notebook fields (mid-chase scenarios) needs to force it
  // open, or the user clicks the chip and sees nothing visibly change.
  const [notebookOpen, setNotebookOpen] = useState(false);
  const [hydrated, setHydrated] = useState(false);

  // Three sequential "pages": pick a scenario -> land here after starting ->
  // once the first ball is recorded, the predictor takes over.
  const stage = !matchStarted ? 'landing' : history.length === 0 ? 'setup' : 'live';

  // Each stage is a full "page" — landing on it scrolled down from wherever
  // the previous page left off (e.g. Setup's Scorer's Notebook can be
  // scrolled well down the page) reads as broken, not just unpolished.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [stage]);

  // Every derived number (balls remaining, runs required, current/required
  // run rate) lives here, computed fresh from the raw inputs on every
  // render — never stored or edited directly, so it can't drift.
  const derived = useMemo(() => deriveMatchStats(inputs), [inputs]);
  const targetIsValid = derived.target !== null && derived.target > 0;

  // The innings can end three ways: the chase is won, the side is bowled
  // out, or the overs run out. Training data (Phase 1) never contains a
  // ball bowled past any of these — Cricsheet's own deliveries stop there —
  // so once one is true the model would be extrapolating into states it's
  // never seen. Ball-recording is gated on this rather than just balls
  // remaining, and the live view surfaces which one actually applies.
  const chaseWon = derived.runsRequired !== null && derived.runsRequired <= 0;
  const allOut = derived.cumWickets >= 10;
  const oversComplete = derived.ballsRemaining <= 0;
  const inningsOver = chaseWon || allOut || oversComplete;
  const inningsOverReason = chaseWon
    ? 'Target reached — the chase is won.'
    : allOut
      ? 'All out — the innings ends there.'
      : oversComplete
        ? 'Overs complete — the innings ends there.'
        : null;

  // Only substitutes into headings when a real team was actually picked —
  // reusing a generic "the batting team" fallback string directly inside a
  // Title Case heading reads badly ("the batting team Won"), so each
  // heading below keeps its own original generic wording when no team is
  // selected, and only swaps in the real name when one is.
  const battingTeamName = inputs.batting_team_name || null;

  // What the header's stage indicator shows — a superset of `stage` that
  // splits "live" into "live" (still in play) vs "result" (decided), so the
  // header actually reflects that something changed once the chase ends
  // instead of sitting on "Live" for the rest of the session.
  const headerStage = stage === 'live' && inningsOver ? 'result' : stage;

  // Restore an in-progress chase after a refresh, once, on mount. Landing
  // back on the marketing page after an accidental Cmd+R is a bad surprise
  // when there was a live simulation running a second ago.
  useEffect(() => {
    try {
      const raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        const saved = JSON.parse(raw);
        if (saved && typeof saved === 'object') {
          if (saved.matchStarted) setMatchStarted(true);
          if (saved.inputs) setInputs(saved.inputs);
          if (Array.isArray(saved.history)) setHistory(saved.history);
        }
      }
    } catch (err) {
      console.warn('Could not restore previous session:', err);
    } finally {
      setHydrated(true);
    }
    // Restore-once-on-mount is intentional — this isn't meant to re-run.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persist on every change, but only once the initial restore above has
  // finished — otherwise the very first render (still holding the empty
  // default state) would overwrite a session worth restoring before it
  // ever gets read.
  useEffect(() => {
    if (!hydrated) return;
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ matchStarted, inputs, history }));
    } catch {
      // Private browsing / storage full / disabled — losing persistence
      // here just means refresh behaves like it used to. Not worth surfacing.
    }
  }, [hydrated, matchStarted, inputs, history]);

  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_BASE}/health_check`);
        if (response.ok) {
          const data = await response.json();
          if (data.status === 'ok') { setBackendConnected(true); return; }
        }
        setBackendConnected(false);
      } catch {
        setBackendConnected(false);
      }
    };
    checkConnection();
  }, []);

  // Almanack Notes numbers (deliveries trained on, AUC, ECE) come from
  // model_stats.json — a static file the training pipeline writes (see
  // phase3_modeling.py / phase4_calibration.py) and that gets copied into
  // frontend/public/ alongside the calibration/event-impact PNGs. It's a
  // same-origin static asset like those PNGs, not an API call, so no
  // API_BASE prefix here. cache: 'no-store' avoids the same "still showing
  // the old chart after a retrain" staleness the PNGs can hit on a
  // browser that's cached the previous bytes under the same filename.
  const [almanackStats, setAlmanackStats] = useState(null);
  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetch('/model_stats.json', { cache: 'no-store' });
        if (!response.ok) throw new Error('model_stats.json not found');
        setAlmanackStats(await response.json());
      } catch {
        setAlmanackStats(null);
      }
    };
    loadStats();
  }, []);

  const almanackDisplayStats = almanackStats ? [
    { label: 'Deliveries Trained On', value: almanackStats.deliveries_trained_on / 1000, decimals: 0, suffix: 'K' },
    { label: 'AUC (Main Model)', value: almanackStats.main_model_auc, decimals: 2, suffix: '' },
    { label: 'Calibration Error (ECE)', value: almanackStats.main_model_ece, decimals: 3, suffix: '' },
  ] : [];

  // Real teams for the Setup page's team-select dropdowns — see
  // phase2_team_priors.py's compute_current_team_snapshot. Each entry is a
  // leakage-safe rolling win-rate prior (same feature the model was
  // actually trained on), not a made-up number. Same fetch pattern as
  // model_stats.json: same-origin static file, no API_BASE, no-store to
  // avoid stale-after-retrain caching.
  const [teamPriors, setTeamPriors] = useState({});
  useEffect(() => {
    const loadTeams = async () => {
      try {
        const response = await fetch('/team_priors.json', { cache: 'no-store' });
        if (!response.ok) throw new Error('team_priors.json not found');
        setTeamPriors(await response.json());
      } catch {
        setTeamPriors({});
      }
    };
    loadTeams();
  }, []);
  const teamNames = useMemo(() => Object.keys(teamPriors).sort(), [teamPriors]);

  // Selecting a team prefills its prior into the (still freely editable)
  // slider — a starting point, not a lock. Dragging the slider afterwards
  // is a legitimate "what if this team were playing above/below their
  // recent form" override; the team name stays attached for the
  // explanation prompt either way.
  const handleTeamSelect = (side, teamName) => {
    setInputs((prev) => {
      const next = { ...prev, [`${side}_team_name`]: teamName };
      if (teamName && teamPriors[teamName]) {
        next[`${side}_team_prior`] = teamPriors[teamName].prior;
      }
      return next;
    });
    if (teamName) setNotebookOpen(true);
  };

  const triggerPrediction = async (currentInputs, prevProba, preBallInputs, rawEvent) => {
    setLoading(true);
    setError(null);

    const stats = deriveMatchStats(currentInputs);
    const recentRunRate = currentInputs.recent_run_rate ?? stats.currentRunRate;

    // "Before this ball" snapshot, purely for the explanation prompt — lets
    // it cite only what actually changed instead of guessing from a single
    // end-state snapshot (see phase6a_explanation.py; this is the fix for
    // commentary citing static, unchanged numbers like wickets-down as if
    // they explained a swing they had nothing to do with).
    const beforeStats = preBallInputs ? deriveMatchStats(preBallInputs) : null;
    const overJustCompleted = !!(
      beforeStats && stats.ballsRemaining < beforeStats.ballsRemaining && stats.ballsBowled % 6 === 0
    );
    // How many legal deliveries actually happened since the last time a
    // prediction was requested — normally 1, but can be more than that
    // (Auto-predict toggled off for a few balls then back on, or a big
    // manual edit via Alter State) and 0 for a wide/no-ball, which
    // doesn't consume a legal delivery. Without this, the explanation
    // prompt has no way to know it's describing a multi-ball jump instead
    // of one discrete ball — which is exactly how you get a sentence
    // confidently narrating "another wicket" as if it alone explains a
    // swing that's actually the combined effect of five or six deliveries.
    // Can go negative if Alter State moves the innings BACKWARD (the user
    // corrects an overs/balls value to something earlier than it was) —
    // the API rejects negative balls_elapsed outright (it isn't a
    // meaningful "how many balls just happened" count), so send null
    // instead of a value that would 422 the whole request. null already
    // means "unknown, describe generically" to the explanation layer.
    const rawBallsElapsed = beforeStats ? beforeStats.ballsRemaining - stats.ballsRemaining : null;
    const ballsElapsed = rawBallsElapsed !== null && rawBallsElapsed >= 0 ? rawBallsElapsed : null;

    // Only the fields /predict actually needs. Deliberately does NOT
    // include anything explanation-only (team names, before/after
    // snapshot, raw_event) — /predict is a fast, local-model call that
    // must never wait on anything else. See phase6b_api.py's comment on
    // /predict for why this is split from /explain below: bundling both
    // into one request meant ANY Groq slowness (a third-party network
    // call, occasionally several seconds under Groq's free-tier rate
    // limit) froze the ENTIRE UI, including the probability number that
    // has nothing to do with it.
    const predictPayload = {
      cum_runs: stats.cumRuns,
      cum_wickets: stats.cumWickets,
      balls_remaining: stats.ballsRemaining,
      runs_required: stats.runsRequired ?? 0,
      required_run_rate: stats.requiredRunRate ?? 0,
      current_run_rate: parseFloat(stats.currentRunRate.toFixed(2)),
      recent_run_rate: parseFloat(recentRunRate.toFixed(2)),
      batting_team_prior: parseFloat(currentInputs.batting_team_prior),
      bowling_team_prior: parseFloat(currentInputs.bowling_team_prior),
      event_type: currentInputs.event_type,
      previous_proba: prevProba,
    };

    let result;
    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predictPayload),
      });
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`API Error: ${response.status} - ${errText}`);
      }
      result = await response.json();
    } catch (err) {
      console.error(err);
      setError(err.message || 'Failed to submit prediction request.');
      setLoading(false);
      return;
    }

    // Stable per-ball id so the /explain response (which resolves later,
    // asynchronously, below) can find and patch the RIGHT history entry —
    // ballIndex/array position isn't safe to rely on here since Undo can
    // remove entries before that response comes back.
    const recordId = ++recordIdCounter.current;

    const newRecord = {
      id: recordId,
      ballIndex: history.length + 1,
      ...predictPayload,
      win_probability: result.win_probability,
      swing: result.swing !== undefined ? result.swing : null,
      // Starts null and gets patched in-place once /explain resolves
      // below — the whole point of the split is that the probability
      // above is already final and on-screen well before this arrives.
      explanation: null,
      // Distinct from "explanation is null because there's nothing to
      // say yet" (the very first ball, which never gets commentary) vs.
      // "still waiting on Groq" vs. "Groq call finished/failed" — the
      // commentary panel below uses this to show a distinct "generating"
      // state instead of jumping straight to the empty-state placeholder.
      explanationPending: prevProba !== null,
      // Snapshot of the raw inputs right before this ball was applied —
      // lets "Undo" restore the exact prior state instead of trying to
      // algebraically reverse an event (which isn't always invertible,
      // e.g. "other_runs" covers 1/2/3/wide/noball indistinguishably).
      preBallInputs: preBallInputs ?? currentInputs,
      // The literal button clicked (dot_ball/single/.../wicket/wide/
      // noball), kept separate from `event_type` above — that field is
      // already collapsed to the model's 5 categories (1/2/3/wide/noball
      // all become "other_runs"), which loses exactly the detail a
      // scorecard strip needs to show. Falls back to a "state set"
      // marker for entries that came from the Setup/Alter-State forms
      // rather than an actual recorded delivery.
      displayEvent: rawEvent ?? 'state_set',
    };
    setHistory((prev) => [...prev, newRecord]);
    setInputs(prev => ({ ...prev, recent_run_rate: recentRunRate, event_type: currentInputs.event_type }));
    setLoading(false);

    // Commentary only ever makes sense once there's a previous probability
    // to swing away from — the very first ball of an innings has nothing
    // to compare against, same gating /predict used to apply internally
    // before this split. Fired here WITHOUT awaiting — the probability
    // above is already committed to history and on screen; letting this
    // resolve in the background, on its own timeline, is the entire
    // point of splitting it out of /predict.
    if (prevProba !== null) {
      const explainPayload = {
        event_type: currentInputs.event_type,
        proba_before: prevProba,
        proba_after: result.win_probability,
        swing: result.swing,
        cum_runs: stats.cumRuns,
        cum_wickets: stats.cumWickets,
        balls_remaining: stats.ballsRemaining,
        runs_required: stats.runsRequired ?? 0,
        required_run_rate: stats.requiredRunRate ?? 0,
        batting_team: currentInputs.batting_team_name || null,
        bowling_team: currentInputs.bowling_team_name || null,
        cum_runs_before: beforeStats ? beforeStats.cumRuns : null,
        cum_wickets_before: beforeStats ? beforeStats.cumWickets : null,
        balls_remaining_before: beforeStats ? beforeStats.ballsRemaining : null,
        required_run_rate_before: beforeStats && beforeStats.requiredRunRate !== null ? beforeStats.requiredRunRate : null,
        over_just_completed: overJustCompleted,
        balls_elapsed: ballsElapsed,
        raw_event: rawEvent || null,
      };
      fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(explainPayload),
      })
        .then((res) => (res.ok ? res.json() : Promise.reject(new Error(`Explain API error: ${res.status}`))))
        .then((data) => {
          setHistory((prev) => prev.map((item) =>
            item.id === recordId ? { ...item, explanation: data.explanation || null, explanationPending: false } : item
          ));
        })
        .catch((err) => {
          // A failed/slow/rate-limited commentary call degrades to no
          // commentary for this one ball — this deliberately never
          // touches the global `error` banner, which is reserved for
          // prediction failures. A missed line of commentary shouldn't
          // read as an app error to the person using it.
          console.warn('Commentary generation failed for this ball:', err);
          setHistory((prev) => prev.map((item) =>
            item.id === recordId ? { ...item, explanationPending: false } : item
          ));
        });
    }
  };

  // Entering the chase just opens the setup stage — no prediction fires
  // until a ball is actually recorded, so the simulator gets its own moment.
  const handleStartChase = () => {
    setMatchStarted(true);
  };

  const handleManualSubmit = async (e) => {
    e.preventDefault();
    if (!targetIsValid || inningsOver) return;
    const lastProba = history.length > 0 ? history[history.length - 1].win_probability : null;
    // Use the state captured when the editor was opened, not `inputs`
    // itself — see preEditInputsRef above. Falls back to `inputs` (old
    // behavior) only if the ref was somehow never set, e.g. the very
    // first ball of the innings being entered via Setup's Scorer's
    // Notebook rather than the live Alter State editor, where there's no
    // real "before" to diff against anyway.
    await triggerPrediction(inputs, lastProba, preEditInputsRef.current ?? inputs);
    setEditingState(false);
  };

  const handleInputChange = (field, value) => {
    setInputs((prev) => ({ ...prev, [field]: value }));
  };

  const handlePresetClick = async (event) => {
    if (!targetIsValid || inningsOver) return;
    const isIllegalDelivery = event === 'wide' || event === 'noball';

    let runsAdded = 0, wicketsAdded = 0;
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

    const newCumRuns = derived.cumRuns + runsAdded;
    const newCumWickets = Math.min(10, derived.cumWickets + wicketsAdded);

    // Only legal deliveries advance the over count — same rule Phase 1 uses
    // when building the training data, so live play and training stay
    // consistent with each other.
    let newOversCompleted = derived.oversCompleted;
    let newBallsThisOver = derived.ballsThisOver;
    if (!isIllegalDelivery) {
      const newBallsBowled = derived.ballsBowled + 1;
      newOversCompleted = Math.floor(newBallsBowled / 6);
      newBallsThisOver = newBallsBowled % 6;
    }

    // Recent form is a smoothed proxy (no full ball-by-ball history is kept
    // for a state the user typed in from scratch), nudged toward this
    // ball's instantaneous rate — same idea as before, just no longer a
    // field the user has to fill in themselves.
    const instantRR = runsAdded * 6;
    const priorRecentRunRate = inputs.recent_run_rate ?? derived.currentRunRate;
    const newRecentRunRate = isIllegalDelivery ? priorRecentRunRate : parseFloat((priorRecentRunRate * 0.8 + instantRR * 0.2).toFixed(2));
    const apiEventType = ['single', 'two', 'three', 'wide', 'noball'].includes(event) ? 'other_runs' : event;

    const progressedInputs = {
      ...inputs, cum_runs: newCumRuns, cum_wickets: newCumWickets,
      overs_completed: newOversCompleted, balls_this_over: newBallsThisOver,
      recent_run_rate: newRecentRunRate, event_type: apiEventType,
    };
    const preBallInputs = inputs;
    setInputs(progressedInputs);

    // The very first ball of the innings always submits (it's what promotes
    // setup -> live); after that, auto-predict governs it.
    if (autoSubmit || history.length === 0) {
      const lastProba = history.length > 0 ? history[history.length - 1].win_probability : null;
      await triggerPrediction(progressedInputs, lastProba, preBallInputs, event);
    }
  };

  // Undo only ever reverts the most recent SUBMITTED ball (i.e. one that
  // made it into `history`). If auto-predict is off and a few taps have
  // advanced `inputs` further without submitting yet, those aren't tracked
  // by history and Undo won't touch them — narrower than "undo my last
  // tap", but unambiguous and never loses a prediction silently.
  const handleUndo = () => {
    if (history.length === 0 || loading) return;
    const lastBall = history[history.length - 1];
    setInputs(lastBall.preBallInputs);
    setHistory((prev) => prev.slice(0, -1));
    setError(null);
  };

  // Reset innings drops history back to zero, which naturally sends the UI
  // back to the setup stage — no separate "are you sure" flow needed.
  const handleReset = () => {
    setInputs(EMPTY_INPUTS);
    setHistory([]);
    setError(null);
    setEditingState(false);
  };

  const handleBackToStart = () => {
    setMatchStarted(false);
    setInputs(EMPTY_INPUTS);
    setHistory([]);
    setError(null);
    setEditingState(false);
  };

  const chartData = useMemo(() => {
    return history.map((item) => {
      const ballsBowled = 120 - item.balls_remaining;
      const overs = Math.floor(ballsBowled / 6);
      const ballInOver = ballsBowled % 6;
      return {
        label: `${overs}.${ballInOver}`,
        probability: Math.round(item.win_probability * 100),
        runs: item.cum_runs, wickets: item.cum_wickets, event: item.event_type, explanation: item.explanation,
      };
    });
  }, [history]);

  const latestPrediction = history[history.length - 1];
  const currentProbability = latestPrediction ? latestPrediction.win_probability : null;
  const currentSwing = latestPrediction ? latestPrediction.swing : null;
  const currentExplanation = latestPrediction ? latestPrediction.explanation : null;
  // True only while /explain is actually in flight for the LATEST ball —
  // distinct from "no explanation, never will be" (first ball) or "the
  // call finished/failed" (explanationPending is set back to false in
  // both of those cases in triggerPrediction). Drives a "generating…"
  // message instead of jumping straight to the empty-state placeholder.
  const currentExplanationPending = latestPrediction ? !!latestPrediction.explanationPending : false;
  const animatedProbability = useAnimatedNumber(currentProbability !== null ? currentProbability * 100 : 0, 700);

  // A brief, one-shot shake on the Match Center panel when a wicket falls.
  // Keyed by a nonce (not just event type) so two wickets in a row each
  // get their own shake instead of the second one being a no-op because
  // "nothing changed" as far as React's concerned.
  const [flashEvent, setFlashEvent] = useState(null);
  const prevHistoryLengthRef = useRef(0);
  useEffect(() => {
    if (history.length > prevHistoryLengthRef.current) {
      const last = history[history.length - 1];
      if (last?.displayEvent === 'wicket') {
        setFlashEvent({ nonce: `${history.length}-${Date.now()}` });
      }
    }
    prevHistoryLengthRef.current = history.length;
  }, [history]);
  const shakeControls = useAnimationControls();
  useEffect(() => {
    if (flashEvent) {
      shakeControls.start({ x: [0, -9, 9, -7, 7, -4, 4, 0], transition: { duration: 0.5, ease: 'easeInOut' } });
    }
  }, [flashEvent, shakeControls]);

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-brand-card border-2 border-brand-pine/20 p-3 rounded-md shadow-xl text-xs max-w-xs font-sans">
          <p className="font-scoreboard font-semibold text-brand-pine/60 mb-1">Over {data.label}</p>
          <p className="text-brand-pine font-bold text-sm mb-1 font-scoreboard">Win Probability: {data.probability}%</p>
          <p className="text-brand-pine/80 font-medium">Score: {data.runs}/{data.wickets}</p>
          <p className="text-brand-pine/80">Event: <span className="capitalize font-semibold text-brand-amber">{data.event.replace('_', ' ')}</span></p>
          {data.explanation && <p className="text-brand-pine/70 mt-2 border-t border-brand-pine/10 pt-1.5 leading-relaxed italic text-[11px]">"{data.explanation}"</p>}
        </div>
      );
    }
    return null;
  };

  const formatOvers = (ballsRemaining) => {
    const bowled = 120 - ballsRemaining;
    return `${Math.floor(bowled / 6)}.${bowled % 6}`;
  };

  const presetButtonBase = "font-bold rounded text-xs transition duration-150 disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer border-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine focus-visible:ring-offset-2 focus-visible:ring-offset-brand-parchment active:scale-95";

  // One-click starting points for the setup page — each is a real, sane
  // match state (checked against deriveMatchStats: none of them land on an
  // already-decided chase), meant to remove the "type five numbers before
  // I can even see the demo work" friction. Filling notebook fields still
  // goes through handleInputChange's normal path, so nothing here bypasses
  // validation — it's just a batch prefill.
  const QUICK_START_SCENARIOS = [
    { key: 'fresh', label: 'Fresh Start', desc: 'Chase begins now', accent: 'border-t-brand-pine/40', fields: { target: '165', cum_runs: 0, cum_wickets: 0, overs_completed: 0, balls_this_over: 0 } },
    { key: 'comfortable', label: 'Comfortable Chase', desc: '50 off 48, 8 wkts in hand', accent: 'border-t-brand-moss', fields: { target: '150', cum_runs: 100, cum_wickets: 2, overs_completed: 12, balls_this_over: 0 } },
    { key: 'uphill', label: 'Uphill Battle', desc: '130 off 60, 6 down', accent: 'border-t-brand-wicket', fields: { target: '190', cum_runs: 60, cum_wickets: 6, overs_completed: 10, balls_this_over: 0 } },
    { key: 'nailbiter', label: 'Nail-Biter Finish', desc: '15 off 8, 4 wkts left', accent: 'border-t-brand-amber', fields: { target: '180', cum_runs: 165, cum_wickets: 6, overs_completed: 18, balls_this_over: 4 } },
  ];

  const handleQuickStart = (scenario) => {
    setInputs((prev) => ({ ...prev, ...scenario.fields }));
    // Only pop the notebook open for scenarios that actually populate it —
    // "Fresh Start" only touches the Target field, which is already visible.
    if (scenario.key !== 'fresh') setNotebookOpen(true);
  };

  const eventButtons = [
    { key: 'dot_ball', label: 'Dot', cls: 'bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine' },
    { key: 'single', label: '1 Run', cls: 'bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine' },
    { key: 'two', label: '2 Runs', cls: 'bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine' },
    { key: 'three', label: '3 Runs', cls: 'bg-brand-parchment-dim hover:bg-brand-pine/10 border-brand-pine/20 text-brand-pine' },
    { key: 'four', label: 'Four', cls: 'bg-brand-moss/10 hover:bg-brand-moss/20 border-brand-moss/40 text-brand-moss' },
    { key: 'six', label: 'Six', cls: 'bg-brand-moss/15 hover:bg-brand-moss/25 border-brand-moss/50 text-brand-moss font-black' },
    { key: 'wicket', label: 'Wicket', cls: 'bg-brand-wicket/10 hover:bg-brand-wicket/20 border-brand-wicket/40 text-brand-wicket' },
    { key: 'wide', label: 'Wide', cls: 'bg-brand-amber/10 hover:bg-brand-amber/20 border-brand-amber/40 text-brand-amber' },
    { key: 'noball', label: 'No Ball', cls: 'bg-brand-amber/10 hover:bg-brand-amber/20 border-brand-amber/40 text-brand-amber' },
  ];

  // Small circular chip per recorded ball, for the scorecard strip — reuses
  // the same color language as eventButtons above (moss for boundaries,
  // wicket-red, amber for extras) so the strip reads as an extension of the
  // input buttons rather than a separate visual system.
  const BALL_CHIP = {
    dot_ball: { label: '•', cls: 'bg-brand-pine/10 text-brand-pine/50 border-brand-pine/10' },
    single: { label: '1', cls: 'bg-brand-parchment-dim text-brand-pine border-brand-pine/15' },
    two: { label: '2', cls: 'bg-brand-parchment-dim text-brand-pine border-brand-pine/15' },
    three: { label: '3', cls: 'bg-brand-parchment-dim text-brand-pine border-brand-pine/15' },
    four: { label: '4', cls: 'bg-brand-moss/15 text-brand-moss border-brand-moss/40 font-black' },
    six: { label: '6', cls: 'bg-brand-moss text-brand-parchment border-brand-moss font-black' },
    wicket: { label: 'W', cls: 'bg-brand-wicket text-brand-parchment border-brand-wicket font-black' },
    wide: { label: 'wd', cls: 'bg-brand-amber/15 text-brand-amber border-brand-amber/40 text-[9px]' },
    noball: { label: 'nb', cls: 'bg-brand-amber/15 text-brand-amber border-brand-amber/40 text-[9px]' },
    // Entries from "Begin Chase With This State" / "Apply Altered State" —
    // not an actual delivery, so a distinct, muted marker rather than
    // pretending it was a ball.
    state_set: { label: '◆', cls: 'bg-brand-pine/5 text-brand-pine/30 border-brand-pine/10 text-[9px]' },
  };
  // Only legal deliveries close out an over (wides/no-balls/state-set
  // entries don't), matching the exact rule Phase 1 uses when building the
  // training data. Anchored to the ACTUAL innings position via
  // balls_remaining (which deriveMatchStats always computes relative to
  // the full 120-ball innings, not relative to when recording started) —
  // not a count of "6 balls since we started recording". Without this,
  // starting mid-innings (Scorer's Notebook, a Quick Start scenario) put
  // the divider 6 *recorded* balls in, which almost never lines up with
  // where the over boundary actually falls once you started partway
  // through an over.
  const scorecardChips = useMemo(() => {
    return history.map((item) => {
      const isLegal = LEGAL_BALL_EVENTS.has(item.displayEvent);
      const ballsBowledInInnings = 120 - item.balls_remaining;
      const overComplete = isLegal && ballsBowledInInnings > 0 && ballsBowledInInnings % 6 === 0;
      return { ...item, overComplete };
    });
  }, [history]); // LEGAL_BALL_EVENTS is a stable module-level constant, not a real dependency

  const StepPip = ({ id, label }) => (
    <div className="relative px-3 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-wide">
      {headerStage === id && (
        <motion.div layoutId="stage-pill" className="absolute inset-0 bg-brand-pine rounded-full" transition={{ type: 'spring', stiffness: 350, damping: 30 }} />
      )}
      <span className={`relative z-10 ${headerStage === id ? 'text-brand-parchment' : 'text-brand-pine/40'}`}>{label}</span>
    </div>
  );

  return (
    <div className="min-h-screen bg-brand-parchment text-brand-pine flex flex-col font-sans">

      {/* Masthead */}
      <header className="border-b-4 border-brand-pine px-6 py-5 flex flex-wrap justify-between items-end gap-4 relative">
        <div className="flex items-center gap-3">
          <motion.div whileHover={{ rotate: 8, scale: 1.06 }} transition={{ type: 'spring', stiffness: 300 }}>
            <LogoMark size={38} />
          </motion.div>
          <div className="flex items-baseline gap-2">
            <h1 className="text-3xl font-display font-black tracking-tight text-brand-pine">Crickcast</h1>
            <span className="text-xs font-scoreboard font-bold px-2 py-0.5 rounded bg-brand-yellow text-brand-pine-dark border border-brand-pine/20">T20</span>
          </div>
        </div>
        <p className="text-xs text-brand-pine/60 font-medium italic font-display">A ball-by-ball win probability engine, with reasoning</p>

        <div className="flex items-center gap-3 ml-auto">
          {matchStarted && (
            <div className="hidden md:flex items-center gap-1 bg-brand-card p-1 rounded-full border-2 border-brand-pine/15">
              <StepPip id="setup" label="Setup" />
              <StepPip id="live" label="Live" />
              <StepPip id="result" label="Result" />
            </div>
          )}
          <div className="flex items-center gap-2 bg-brand-card px-3 py-1.5 rounded-full border-2 border-brand-pine/15">
            {backendConnected === null ? (
              <><div className="w-2 h-2 rounded-full bg-brand-amber animate-pulse" /><span className="text-xs text-brand-pine/60 font-medium">Connecting…</span></>
            ) : backendConnected ? (
              <><Wifi className="w-3.5 h-3.5 text-brand-moss" aria-hidden="true" /><span className="text-xs text-brand-moss font-semibold uppercase tracking-wide">Live</span></>
            ) : (
              <>
                <WifiOff className="w-3.5 h-3.5 text-brand-wicket" aria-hidden="true" />
                <span className="text-xs text-brand-wicket font-semibold uppercase tracking-wide">Offline</span>
                <button onClick={() => window.location.reload()} className="text-[10px] text-brand-pine/60 hover:text-brand-pine underline ml-1 cursor-pointer">Reconnect</button>
              </>
            )}
          </div>
          {matchStarted && (
            <button onClick={handleBackToStart} className="flex items-center gap-1.5 text-xs text-brand-pine hover:text-brand-pine-dark bg-brand-card hover:bg-brand-parchment-dim px-3 py-1.5 rounded-md border-2 border-brand-pine/15 hover:border-brand-pine/30 transition-all cursor-pointer font-semibold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine">
              <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
              Start Over
            </button>
          )}
        </div>
        <div className="header-shimmer absolute bottom-0 left-0 right-0 h-1" />
      </header>

      <AnimatePresence mode="wait">
        {stage === 'landing' && (
          /* ---------- PAGE 1 — LANDING ---------- */
          <motion.main
            key="landing"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.35 }}
            className="flex-1 flex items-center justify-center relative overflow-hidden px-6 py-16"
          >
            <div className="drift-blob pointer-events-none absolute -top-20 -left-20 w-80 h-80 rounded-full bg-brand-yellow/20 blur-3xl" />
            <div className="drift-blob pointer-events-none absolute bottom-0 right-0 w-96 h-96 rounded-full bg-brand-moss/15 blur-3xl" style={{ animationDelay: '3s' }} />
            <div className="drift-blob pointer-events-none absolute top-1/3 right-1/4 w-64 h-64 rounded-full bg-brand-wicket/5 blur-3xl" style={{ animationDelay: '5.5s' }} />

            <motion.div
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}
              className="relative z-10 text-center max-w-xl"
            >
              <motion.div
                className="flex justify-center mb-6"
                animate={{ y: [0, -8, 0] }} transition={{ duration: 3.2, repeat: Infinity, ease: 'easeInOut' }}
              >
                <LogoMark size={72} />
              </motion.div>
              <h2 className="text-4xl md:text-5xl font-display font-black text-brand-pine mb-4 leading-tight">
                <motion.span className="block" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.2 }}>
                  Watch a T20 chase
                </motion.span>
                <motion.span className="block" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.32 }}>
                  unfold, ball by ball.
                </motion.span>
              </h2>
              <motion.p
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.45 }}
                className="text-sm text-brand-pine/70 mb-8 leading-relaxed"
              >
                A model trained on 3,000+ real matches predicts the chasing team's win probability
                after every delivery — with live AI commentary explaining why it moved.
              </motion.p>
              <motion.button
                initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.55 }}
                whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
                onClick={handleStartChase}
                className="group relative inline-flex items-center gap-2 overflow-hidden bg-brand-pine hover:bg-brand-pine-dark text-brand-parchment font-display font-bold py-3.5 px-8 rounded-md text-base shadow-lg cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber focus-visible:ring-offset-2 focus-visible:ring-offset-brand-parchment"
              >
                <span className="pointer-events-none absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 ease-out bg-gradient-to-r from-transparent via-white/20 to-transparent" aria-hidden="true" />
                <span className="relative">Start the Chase</span> <ArrowRight className="relative w-4 h-4" aria-hidden="true" />
              </motion.button>
              <motion.p
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5, delay: 0.65 }}
                className="text-[11px] text-brand-pine/40 mt-4"
              >
                Next: set the scene, then bowl the first ball.
              </motion.p>
            </motion.div>
          </motion.main>
        )}

        {stage === 'setup' && (
          /* ---------- PAGE 2 — SET THE STAGE (simulator, standalone) ---------- */
          <motion.main
            key="setup"
            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.4 }}
            className="flex-1 relative overflow-hidden px-6 py-12 md:py-16 flex flex-col items-center"
          >
            <div className="drift-blob pointer-events-none absolute -top-24 right-[-6rem] w-96 h-96 rounded-full bg-brand-yellow/15 blur-3xl" />
            <div className="drift-blob pointer-events-none absolute bottom-[-6rem] left-[-6rem] w-80 h-80 rounded-full bg-brand-moss/10 blur-3xl" style={{ animationDelay: '2s' }} />

            <motion.div
              initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.1 }}
              className="relative z-10 w-full max-w-4xl text-center mb-8"
            >
              <div className="inline-flex items-center gap-2 text-[11px] font-bold uppercase tracking-widest text-brand-amber mb-3">
                <Flag className="w-3.5 h-3.5" aria-hidden="true" /> Set the stage
              </div>
              <h2 className="text-3xl md:text-4xl font-display font-black text-brand-pine mb-3">Set the target, then bowl the first ball.</h2>
              <p className="text-sm text-brand-pine/60 max-w-lg mx-auto leading-relaxed">
                Crickcast can't predict anything until it knows the target — everything else on this
                page (score, run rates, balls left) is calculated from what you enter below, not the
                other way around.
              </p>
            </motion.div>

            {/* Quick Start — one click fills the fields below with a real, sane
                match state, purely as a shortcut. Nothing here is submitted
                automatically; it's still the user who hits a ball button or
                "Begin Chase" afterwards. */}
            <motion.div
              initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.13 }}
              className="relative z-10 w-full max-w-4xl mb-6"
            >
              <p className="text-[10px] uppercase font-bold tracking-widest text-brand-pine/40 mb-2.5 text-center">Quick start — or fill in your own below</p>
              <div className="flex flex-wrap justify-center gap-2.5">
                {QUICK_START_SCENARIOS.map((scenario, i) => (
                  <motion.button
                    key={scenario.key} type="button" onClick={() => handleQuickStart(scenario)}
                    initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.15 + i * 0.05 }}
                    whileHover={{ y: -3 }} whileTap={{ scale: 0.97 }}
                    className={`text-left bg-brand-card hover:bg-white border border-brand-pine/10 border-t-2 ${scenario.accent} rounded-md px-3.5 py-2.5 shadow-sm hover:shadow-md transition-shadow cursor-pointer`}
                  >
                    <p className="text-xs font-display font-bold text-brand-pine">{scenario.label}</p>
                    <p className="text-[10px] text-brand-pine/50">{scenario.desc}</p>
                  </motion.button>
                ))}
              </div>
            </motion.div>

            {/* Quiet preview of the scenario about to start — every value here is
                derived from the inputs below, and shows "—" until there's enough
                to compute it. Not editable directly. */}
            <motion.div
              initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.18 }}
              className="relative z-10 w-full max-w-4xl grid grid-cols-3 md:grid-cols-6 gap-2 mb-8"
            >
              {[
                { label: 'Target', value: derived.target !== null ? derived.target : '—' },
                { label: 'Score', value: `${derived.cumRuns}/${derived.cumWickets}` },
                { label: 'Overs', value: formatOvers(derived.ballsRemaining) },
                { label: 'Balls Left', value: derived.target !== null ? derived.ballsRemaining : '—' },
                { label: 'Needed', value: derived.runsRequired !== null ? derived.runsRequired : '—' },
                { label: 'Req. R.R', value: derived.requiredRunRate !== null ? derived.requiredRunRate.toFixed(1) : '—' },
              ].map((stat) => (
                <div key={stat.label} className="bg-brand-card border-2 border-brand-pine/10 rounded-md py-2 text-center">
                  <p className="text-[8px] uppercase font-bold tracking-widest text-brand-pine/40 mb-0.5">{stat.label}</p>
                  <p className="text-sm font-scoreboard font-bold text-brand-pine">{stat.value}</p>
                </div>
              ))}
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, delay: 0.25 }}
              className="relative z-10 w-full max-w-4xl bg-brand-card border-2 border-brand-pine/15 rounded-lg p-5 md:p-7 shadow-lg"
            >
              {/* Target is required regardless of which path below the user takes,
                  so it lives outside the (optional) Scorer's Notebook. */}
              <div className="mb-6 pb-6 border-b-2 border-brand-pine/10">
                <label htmlFor="target-input" className="block text-sm font-display font-bold text-brand-pine mb-1">
                  What's the target? <span className="text-brand-wicket" aria-hidden="true">*</span>
                </label>
                <p className="text-xs text-brand-pine/50 mb-2.5">First innings total + 1. Required — nothing below works without it.</p>
                <input
                  id="target-input" type="number" min="1" placeholder="e.g. 165"
                  value={inputs.target}
                  onChange={(e) => handleInputChange('target', e.target.value)}
                  className="w-full sm:w-64 bg-brand-card border-2 border-brand-pine/20 rounded py-2.5 px-3 text-brand-pine text-lg font-scoreboard font-bold focus:outline-none focus:border-brand-amber"
                />
              </div>

              {/* Optional real-team selection — prefills the team-strength
                  sliders (inside the Scorer's Notebook below) from each
                  team's actual leakage-safe rolling win rate instead of a
                  bare 50/50 default, and personalizes the Groq commentary.
                  Leaving both on "Custom" behaves exactly as before. */}
              {teamNames.length > 0 && (
                <div className="mb-6 pb-6 border-b-2 border-brand-pine/10 grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="batting-team-select" className="block text-xs font-display font-bold text-brand-pine mb-1.5">Batting team</label>
                    <select
                      id="batting-team-select" value={inputs.batting_team_name}
                      onChange={(e) => handleTeamSelect('batting', e.target.value)}
                      className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm focus:outline-none focus:border-brand-amber"
                    >
                      <option value="">Custom (use sliders below)</option>
                      {teamNames.map((name) => (
                        <option key={name} value={name} disabled={name === inputs.bowling_team_name}>{name}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label htmlFor="bowling-team-select" className="block text-xs font-display font-bold text-brand-pine mb-1.5">Bowling team</label>
                    <select
                      id="bowling-team-select" value={inputs.bowling_team_name}
                      onChange={(e) => handleTeamSelect('bowling', e.target.value)}
                      className="w-full bg-brand-card border-2 border-brand-pine/20 rounded py-2 px-3 text-brand-pine text-sm focus:outline-none focus:border-brand-amber"
                    >
                      <option value="">Custom (use sliders below)</option>
                      {teamNames.map((name) => (
                        <option key={name} value={name} disabled={name === inputs.batting_team_name}>{name}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              <fieldset className="mb-2">
                <legend className="text-xs text-brand-pine/60 mb-3 font-semibold uppercase tracking-wide">Record the first ball:</legend>
                <div className="grid grid-cols-3 md:grid-cols-9 gap-2.5">
                  {eventButtons.map((btn, i) => (
                    <motion.button
                      key={btn.key} type="button"
                      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25, delay: 0.3 + i * 0.03 }}
                      whileHover={{ y: -2 }} whileTap={{ scale: 0.94 }}
                      onClick={() => handlePresetClick(btn.key)}
                      disabled={loading || !targetIsValid || inningsOver}
                      className={`${presetButtonBase} ${btn.cls} py-5`}
                    >
                      {btn.label}
                    </motion.button>
                  ))}
                </div>
                {!targetIsValid && (
                  <p className="text-xs text-brand-amber mt-2.5 font-medium">Enter a target above to start recording deliveries.</p>
                )}
                {targetIsValid && inningsOver && (
                  <p className="text-xs text-brand-amber mt-2.5 font-medium">{inningsOverReason} Open the Scorer's Notebook below to adjust the state.</p>
                )}
              </fieldset>

              {loading && (
                <div className="flex items-center justify-center gap-2 text-xs text-brand-pine/60 mt-4">
                  <div className="w-3.5 h-3.5 border-2 border-brand-pine border-t-transparent rounded-full animate-spin" />
                  Bowling the first delivery…
                </div>
              )}

              {error && (
                <div className="bg-brand-wicket/10 border-2 border-brand-wicket/30 text-brand-wicket p-4 rounded-lg text-xs leading-relaxed flex gap-2 mt-4">
                  <span className="text-sm" aria-hidden="true">⚠</span>
                  <div><p className="font-semibold">Simulation error</p><p className="mt-0.5">{error}</p></div>
                </div>
              )}

              <form onSubmit={handleManualSubmit} className="mt-5">
                <details className="group" open={notebookOpen} onToggle={(e) => setNotebookOpen(e.target.open)}>
                  <summary className="cursor-pointer select-none text-xs font-display font-bold text-brand-pine hover:text-brand-amber mb-3 flex items-center gap-2 list-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-pine rounded">
                    <span className="inline-block transition-transform duration-200 group-open:rotate-90 text-brand-amber">▶</span>
                    Scorer's Notebook — chase already underway?
                    <span className="text-brand-pine/50 font-normal font-sans hidden sm:inline">(optional — leave at 0 for a fresh start)</span>
                  </summary>
                  <ScorersNotebookFields inputs={inputs} onChange={handleInputChange} showEventType={false} />
                  <button type="submit" disabled={loading || !targetIsValid || inningsOver} className="mt-4 w-full md:w-auto bg-brand-pine hover:bg-brand-pine-dark text-brand-parchment font-display font-bold py-3 px-6 rounded-md text-sm transition-all shadow-md flex justify-center items-center gap-2 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber focus-visible:ring-offset-2 focus-visible:ring-offset-brand-parchment">
                    {loading ? (<><div className="w-4 h-4 border-2 border-brand-parchment border-t-transparent rounded-full animate-spin" />Starting…</>) : ('Begin Chase With This State')}
                  </button>
                  {targetIsValid && inningsOver && (
                    <p className="text-xs text-brand-amber mt-2.5 font-medium">{inningsOverReason} Adjust the state above to begin from a point still in play.</p>
                  )}
                </details>
              </form>
            </motion.div>
          </motion.main>
        )}

        {stage === 'live' && (
          /* ---------- PAGE 3 — LIVE PREDICTOR ---------- */
          <motion.main
            key="live"
            initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.35 }}
            className="flex-1 flex flex-col"
          >
            {/* Match Center — full-width hero, own section */}
            <section className="p-6 max-w-[1300px] w-full mx-auto">
              <motion.div animate={shakeControls} className="relative">
                <motion.div
                  initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.4 }}
                  className="relative overflow-hidden bg-brand-pine rounded-lg p-5 md:p-6 shadow-lg scoreboard-panel-glow border-4 border-brand-pine-dark"
                >
                  {inningsOver && chaseWon && <Confetti />}

                <div className="flex items-center gap-2 mb-4">
                  {inningsOver ? (
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-amber" />
                  ) : (
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-yellow opacity-75" />
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-brand-yellow" />
                    </span>
                  )}
                  <span className="text-[10px] font-bold uppercase tracking-widest text-brand-parchment/50">
                    {inningsOver
                      ? `Match Center — ${chaseWon ? (battingTeamName ? `${battingTeamName} Won` : 'Chase Won') : (battingTeamName ? `${battingTeamName} Fell Short` : 'Chase Failed')}`
                      : 'Match Center'}
                  </span>
                </div>

                <div className="grid grid-cols-3 md:grid-cols-6 gap-3 text-center pb-4 border-b-2 border-brand-parchment/10">
                  {[
                    { label: 'Target', value: derived.target },
                    { label: 'Score', value: `${derived.cumRuns}/${derived.cumWickets}` },
                    { label: 'Overs Bowled', value: formatOvers(derived.ballsRemaining) },
                    { label: 'Balls Left', value: derived.ballsRemaining },
                    { label: 'Runs Needed', value: derived.runsRequired },
                    { label: 'Req. R.R', value: derived.requiredRunRate !== null ? derived.requiredRunRate.toFixed(1) : '—' },
                  ].map((stat, i) => (
                    <motion.div
                      key={stat.label}
                      initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: i * 0.05 }}
                      className="bg-brand-pine-dark rounded p-2.5"
                    >
                      <p className="text-[9px] uppercase font-bold tracking-widest text-brand-parchment/50 mb-1 leading-tight">{stat.label}</p>
                      <AnimatePresence mode="popLayout">
                        <motion.p
                          key={stat.value}
                          initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} transition={{ duration: 0.25 }}
                          className="text-lg md:text-xl font-scoreboard font-bold text-brand-yellow scoreboard-glow"
                        >
                          {stat.value}
                        </motion.p>
                      </AnimatePresence>
                    </motion.div>
                  ))}
                </div>

                <div className="flex flex-col md:flex-row justify-between items-center gap-6 py-6 border-b-2 border-brand-parchment/10" aria-live="polite" aria-atomic="true">
                  <div className="flex-1 text-center md:text-left">
                    <p className="text-xs uppercase font-bold tracking-widest text-brand-parchment/50 mb-1">
                      {inningsOver
                        ? `Final Result — ${chaseWon ? 'Target Chased Down' : 'Chase Fell Short'}`
                        : (battingTeamName ? `${battingTeamName} Win Probability` : 'Chasing Team Win Probability')}
                    </p>
                    <div className="inline-flex items-baseline gap-4 mt-1">
                      <span className="text-6xl md:text-7xl font-black font-scoreboard text-brand-yellow scoreboard-glow tracking-tight">
                        {currentProbability !== null ? `${animatedProbability.toFixed(1)}%` : '—'}
                      </span>
                      <AnimatePresence>
                        {currentSwing !== null && (
                          <motion.div
                            key={currentSwing}
                            initial={{ opacity: 0, scale: 0.7 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.7 }} transition={{ duration: 0.25 }}
                            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border-2 ${currentSwing >= 0 ? 'bg-brand-moss/20 text-brand-moss border-brand-moss/40' : 'bg-brand-wicket/20 text-brand-wicket border-brand-wicket/40'}`}
                          >
                            {currentSwing >= 0 ? <TrendingUp className="w-3.5 h-3.5" aria-hidden="true" /> : <TrendingDown className="w-3.5 h-3.5" aria-hidden="true" />}
                            <span className="font-scoreboard">{currentSwing >= 0 ? '+' : '-'}{Math.abs(currentSwing * 100).toFixed(1)}%</span>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </div>
                  </div>
                  <div className="flex gap-4 border-l-2 border-brand-parchment/10 pl-6 h-12 items-center text-xs hidden md:flex">
                    <div><p className="text-[9px] uppercase font-bold text-brand-parchment/40">Current R.R</p><p className="text-sm font-semibold font-scoreboard text-brand-parchment mt-0.5">{derived.currentRunRate.toFixed(1)}</p></div>
                    <div className="h-4 border-r border-brand-parchment/10" />
                    <div><p className="text-[9px] uppercase font-bold text-brand-parchment/40">Required R.R</p><p className="text-sm font-semibold font-scoreboard text-brand-parchment mt-0.5">{derived.requiredRunRate !== null ? derived.requiredRunRate.toFixed(1) : '—'}</p></div>
                  </div>
                </div>

                <div className="mt-5 bg-brand-pine-dark rounded-md p-4" aria-live="polite">
                  <div className="flex items-center gap-2 mb-2 text-xs font-bold uppercase tracking-wider text-brand-yellow">
                    <Sparkles className="w-4 h-4" aria-hidden="true" /><span className="font-display">Commentary</span>
                  </div>
                  <AnimatePresence mode="wait">
                    <motion.p
                      key={currentExplanationPending ? 'pending' : currentExplanation}
                      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }}
                      className="text-sm text-brand-parchment/90 leading-relaxed min-h-[40px] flex items-center gap-2"
                    >
                      {currentExplanationPending ? (
                        <>
                          <span className="inline-block w-3 h-3 border-2 border-brand-parchment/40 border-t-brand-yellow rounded-full animate-spin shrink-0" aria-hidden="true" />
                          <span className="text-brand-parchment/60 italic">Generating commentary…</span>
                        </>
                      ) : (
                        currentExplanation || "Tap a ball event below to generate live commentary explaining each swing in win probability."
                      )}
                    </motion.p>
                  </AnimatePresence>
                </div>
                </motion.div>
              </motion.div>
            </section>

            {/* Control strip — compact, always-available, full width */}
            <section className="px-6 w-full max-w-[1300px] mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35, delay: 0.1 }}
                className="bg-brand-card border-2 border-brand-pine/15 rounded-lg p-4 shadow-sm"
              >
                <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                  <h2 className="text-xs font-display font-bold uppercase tracking-wide text-brand-pine flex items-center gap-2">
                    <Activity className="w-4 h-4 text-brand-amber" aria-hidden="true" />
                    Next Ball
                  </h2>
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2 text-xs">
                      <label className="text-brand-pine/70 cursor-pointer font-medium" htmlFor="auto-submit-toggle">Auto-predict</label>
                      <input id="auto-submit-toggle" type="checkbox" checked={autoSubmit} onChange={(e) => setAutoSubmit(e.target.checked)}
                        className="rounded border-brand-pine/30 text-brand-pine focus:ring-brand-pine w-4 h-4 cursor-pointer" />
                    </div>
                    <button onClick={() => setEditingState((v) => {
                        const next = !v;
                        // Snapshot `inputs` at the moment the editor opens — see
                        // preEditInputsRef above for why this can't just be read
                        // from `inputs` later, at submit time.
                        if (next) preEditInputsRef.current = inputs;
                        return next;
                      })} className={`flex items-center gap-1.5 text-xs font-semibold cursor-pointer ${editingState ? 'text-brand-amber' : 'text-brand-pine/70 hover:text-brand-pine'}`}>
                      <Settings2 className="w-3.5 h-3.5" aria-hidden="true" /> {editingState ? 'Close editor' : 'Alter state'}
                    </button>
                    <button onClick={handleUndo} disabled={history.length === 0 || loading} title={history.length > 0 ? 'Revert the last recorded ball' : 'Nothing to undo yet'}
                      className="flex items-center gap-1.5 text-xs text-brand-pine/70 hover:text-brand-pine font-semibold cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:text-brand-pine/70">
                      <Undo2 className="w-3.5 h-3.5" aria-hidden="true" /> Undo
                    </button>
                    <button onClick={handleReset} className="flex items-center gap-1.5 text-xs text-brand-pine/70 hover:text-brand-pine font-semibold cursor-pointer">
                      <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" /> Reset innings
                    </button>
                  </div>
                </div>

                {inningsOver && (
                  <div className="mb-3 flex items-center gap-2 bg-brand-amber/10 border-2 border-brand-amber/30 text-brand-pine text-xs font-semibold rounded-md px-3 py-2.5">
                    <Flag className="w-3.5 h-3.5 text-brand-amber shrink-0" aria-hidden="true" />
                    {inningsOverReason} No further deliveries can be recorded — reset the innings to run another scenario.
                  </div>
                )}

                <div className="grid grid-cols-5 md:grid-cols-9 gap-2">
                  {eventButtons.map((btn) => (
                    <button key={btn.key} type="button" onClick={() => handlePresetClick(btn.key)} disabled={loading || inningsOver} className={`${presetButtonBase} ${btn.cls} py-3.5`}>
                      {btn.label}
                    </button>
                  ))}
                </div>

                <AnimatePresence>
                  {editingState && (
                    <motion.form
                      onSubmit={handleManualSubmit}
                      initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} transition={{ duration: 0.3 }}
                      className="overflow-hidden"
                    >
                      <div className="pt-4 mt-4 border-t border-brand-pine/10">
                        <ScorersNotebookFields inputs={inputs} onChange={handleInputChange} showEventType />
                        <button type="submit" disabled={loading || inningsOver} className="mt-4 w-full md:w-auto bg-brand-pine hover:bg-brand-pine-dark text-brand-parchment font-display font-bold py-2.5 px-6 rounded-md text-sm transition-all shadow-md flex justify-center items-center gap-2 cursor-pointer disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-amber">
                          {loading ? (<><div className="w-4 h-4 border-2 border-brand-parchment border-t-transparent rounded-full animate-spin" />Applying…</>) : ('Apply Altered State')}
                        </button>
                        {inningsOver && (
                          <p className="text-xs text-brand-amber mt-2 font-medium">{inningsOverReason} Adjust the fields above so the state is still in play before applying.</p>
                        )}
                      </div>
                    </motion.form>
                  )}
                </AnimatePresence>

                {error && (
                  <div className="bg-brand-wicket/10 border-2 border-brand-wicket/30 text-brand-wicket p-3 rounded-lg text-xs leading-relaxed flex gap-2 mt-4">
                    <span aria-hidden="true">⚠</span>
                    <div><p className="font-semibold">Simulation error</p><p className="mt-0.5">{error}</p></div>
                  </div>
                )}
              </motion.div>
            </section>

            {/* Evolution chart — full width, own section, fixed height (fixes the render-at-0px bug) */}
            <section className="p-6 max-w-[1300px] w-full mx-auto">
              <motion.div
                initial={{ opacity: 0, y: 15 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.3 }} transition={{ duration: 0.4 }}
                className="bg-brand-card border-2 border-brand-pine/15 rounded-lg p-5 shadow-md"
              >
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-sm font-display font-bold uppercase tracking-wide text-brand-pine flex items-center gap-2">
                    <Layers className="w-4 h-4 text-brand-amber" aria-hidden="true" /> Probability Through the Innings
                  </h2>
                  <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wide text-brand-moss">
                    <Radio className="w-3 h-3" aria-hidden="true" /> Updating live
                  </div>
                </div>
                <div className="w-full relative h-[320px]">
                  {history.length <= 1 ? (
                    <div className="absolute inset-0 flex flex-col justify-center items-center text-center p-6 text-brand-pine/40 border-2 border-dashed border-brand-pine/15 rounded-lg z-10 bg-brand-card">
                      <p className="text-sm font-medium">The curve builds as you play balls</p>
                      <p className="text-xs text-brand-pine/30 mt-1">Record a couple more deliveries above to see it take shape.</p>
                    </div>
                  ) : null}
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={chartData} margin={{ top: 15, right: 15, left: -20, bottom: 5 }}>
                      <defs>
                        <linearGradient id="probFill" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#f2c230" stopOpacity={0.5} />
                          <stop offset="100%" stopColor="#f2c230" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(27,58,43,0.1)" />
                      <XAxis dataKey="label" stroke="rgba(27,58,43,0.4)" tick={{ fill: '#1b3a2b99', fontSize: 10 }} axisLine={{ stroke: 'rgba(27,58,43,0.2)' }} />
                      <YAxis domain={[0, 100]} stroke="rgba(27,58,43,0.4)" tick={{ fill: '#1b3a2b99', fontSize: 10 }} axisLine={{ stroke: 'rgba(27,58,43,0.2)' }} tickFormatter={(v) => `${v}%`} />
                      <Tooltip content={<CustomTooltip />} />
                      <ReferenceLine y={50} stroke="rgba(27,58,43,0.3)" strokeDasharray="4 4" label={{ value: '50/50', fill: '#1b3a2b66', fontSize: 10, position: 'top' }} />
                      <Area type="monotone" dataKey="probability" stroke="#1b3a2b" strokeWidth={3} fill="url(#probFill)"
                        dot={{ r: 4, stroke: '#f2c230', strokeWidth: 2, fill: '#1b3a2b' }} activeDot={{ r: 6, strokeWidth: 0, fill: '#f2c230' }} animationDuration={500} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {history.length > 0 && (
                  <div className="mt-5 pt-5 border-t-2 border-brand-pine/10">
                    <p className="text-[10px] uppercase font-bold tracking-widest text-brand-pine/40 mb-2.5">This Innings, Ball by Ball</p>
                    <div className="flex flex-wrap gap-2 items-center">
                      {scorecardChips.map((item, i) => {
                        const chip = BALL_CHIP[item.displayEvent] || BALL_CHIP.state_set;
                        return (
                          <React.Fragment key={i}>
                            <motion.span
                              layout
                              initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} transition={{ type: 'spring', stiffness: 400, damping: 20 }}
                              whileHover={{ scale: 1.15, y: -2 }}
                              title={`${item.cum_runs}/${item.cum_wickets} — ${Math.round(item.win_probability * 100)}% win chance`}
                              className={`inline-flex items-center justify-center w-8 h-8 rounded-full border-2 text-[11px] font-scoreboard leading-none shadow-sm cursor-default ${chip.cls}`}
                            >
                              {chip.label}
                            </motion.span>
                            {item.overComplete && i < scorecardChips.length - 1 && (
                              <span className="w-px h-6 bg-brand-pine/15 mx-0.5" aria-hidden="true" />
                            )}
                          </React.Fragment>
                        );
                      })}
                    </div>
                  </div>
                )}
              </motion.div>
            </section>

            {/* Almanack Notes — its own full-bleed section, own visual identity */}
            <motion.section
              initial={{ opacity: 0, y: 15 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.2 }} transition={{ duration: 0.4 }}
              className="bg-brand-pine-dark py-10 px-6 mt-2"
            >
              <div className="max-w-[1300px] w-full mx-auto">
                <div className="flex items-center gap-2 mb-2 border-b-2 border-brand-parchment/10 pb-3">
                  <Info className="w-5 h-5 text-brand-yellow" aria-hidden="true" />
                  <h2 className="text-base font-display font-bold uppercase tracking-wide text-brand-parchment">Almanack Notes — Model Diagnostics</h2>
                </div>

                <div className="grid grid-cols-3 gap-3 my-6">
                  {almanackDisplayStats.length > 0 ? almanackDisplayStats.map((stat, i) => (
                    <motion.div key={stat.label} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.3, delay: i * 0.1 }}
                      className="bg-brand-pine rounded-lg p-4 text-center border border-brand-parchment/10">
                      <p className="text-2xl md:text-3xl font-scoreboard font-bold text-brand-yellow scoreboard-glow">
                        <AnimatedStat value={stat.value} decimals={stat.decimals} suffix={stat.suffix} />
                      </p>
                      <p className="text-[10px] uppercase tracking-widest text-brand-parchment/50 font-bold mt-1">{stat.label}</p>
                    </motion.div>
                  )) : (
                    <div className="col-span-3 bg-brand-pine rounded-lg p-4 text-center border border-brand-parchment/10">
                      <p className="text-xs text-brand-parchment/50">
                        Model stats unavailable — run the training pipeline (phase3/phase4) and copy <code className="text-brand-yellow/80">model_stats.json</code> into <code className="text-brand-yellow/80">frontend/public/</code>.
                      </p>
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <motion.div whileHover={{ y: -3 }} className="bg-brand-card border-2 border-brand-pine/10 rounded-lg p-5 flex flex-col justify-between transition-shadow hover:shadow-lg">
                    <div>
                      <h3 className="text-sm font-display font-bold text-brand-pine mb-1">Probability Calibration Curve</h3>
                      <p className="text-xs text-brand-pine/60 mb-4">Verifies whether the predicted win probabilities match actual historical match outcomes.</p>
                      <div className="aspect-[4/3] bg-white border-2 border-brand-pine/10 rounded flex items-center justify-center overflow-hidden">
                        <img src="/calibration_curve.png" alt="Calibration curve showing predicted vs actual win probability" className="max-h-full max-w-full object-contain p-1"
                          onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span class="text-xs text-brand-pine/40">Calibration curve file missing</span>'; }} />
                      </div>
                    </div>
                    <p className="text-xs text-brand-pine/80 mt-4 leading-relaxed bg-brand-parchment-dim/50 p-3 rounded border border-brand-pine/10">
                      <span className="font-bold text-brand-amber">Reading it:</span> a perfectly calibrated model's line sits on the diagonal. The main model tracks it closely across the full probability range.
                    </p>
                  </motion.div>

                  <motion.div whileHover={{ y: -3 }} className="bg-brand-card border-2 border-brand-pine/10 rounded-lg p-5 flex flex-col justify-between transition-shadow hover:shadow-lg">
                    <div>
                      <h3 className="text-sm font-display font-bold text-brand-pine mb-1">Event-Impact Probability Swing</h3>
                      <p className="text-xs text-brand-pine/60 mb-4">Average win-probability swing per ball, grouped by event type — measured directly from the model's own predictions.</p>
                      <div className="aspect-[4/3] bg-white border-2 border-brand-pine/10 rounded flex items-center justify-center overflow-hidden">
                        <img src="/event_impact_swing.png" alt="Bar chart of average win-probability swing by event type" className="max-h-full max-w-full object-contain p-1"
                          onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span class="text-xs text-brand-pine/40">Event impact file missing</span>'; }} />
                      </div>
                    </div>
                    <p className="text-xs text-brand-pine/80 mt-4 leading-relaxed bg-brand-parchment-dim/50 p-3 rounded border border-brand-pine/10">
                      <span className="font-bold text-brand-amber">Reading it:</span> wickets swing the probability hardest, sixes swing it back the most — exactly what cricket sense would predict.
                    </p>
                  </motion.div>

                  <motion.div whileHover={{ y: -3 }} className="bg-brand-card border-2 border-brand-pine/10 rounded-lg p-5 flex flex-col justify-between transition-shadow hover:shadow-lg">
                    <div>
                      <h3 className="text-sm font-display font-bold text-brand-pine mb-1">Feature Importance</h3>
                      <p className="text-xs text-brand-pine/60 mb-4">What the trained model actually leans on — XGBoost's gain-based importance per feature, straight from the model file.</p>
                      <div className="aspect-[4/3] bg-white border-2 border-brand-pine/10 rounded flex items-center justify-center overflow-hidden">
                        <img src="/feature_importance.png" alt="Bar chart of XGBoost feature importance for the main model" className="max-h-full max-w-full object-contain p-1"
                          onError={(e) => { e.target.style.display = 'none'; e.target.parentNode.innerHTML = '<span class="text-xs text-brand-pine/40">Feature importance file missing</span>'; }} />
                      </div>
                    </div>
                    <p className="text-xs text-brand-pine/80 mt-4 leading-relaxed bg-brand-parchment-dim/50 p-3 rounded border border-brand-pine/10">
                      <span className="font-bold text-brand-amber">Reading it:</span> required run rate dominates by a wide margin — cricket sense again: how hard the equation is right now matters more than any single input feeding it.
                    </p>
                  </motion.div>
                </div>
              </div>
            </motion.section>
          </motion.main>
        )}
      </AnimatePresence>

      <footer className="border-t-2 border-brand-pine/15 py-4 bg-brand-parchment text-center text-xs text-brand-pine/50 font-medium">
        <p>Crickcast — Win Probability Engine, FastAPI + XGBoost.</p>
      </footer>
    </div>
  );
}