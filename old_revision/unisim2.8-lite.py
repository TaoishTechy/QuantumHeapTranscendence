# unisim-3.0-AGI.py
# QuantumHeapTranscendence v3.0 AGI Emergence
# Upgraded by Gemini on June 18, 2025
# Description: A simulation focused on the emergence of AGI within a
# quantum-inspired universe, with a robust curses-based interface and console.

import math
import random
import time
import datetime
import curses
import json
import numpy as np
from collections import defaultdict, deque

# --- Core Constants ---
PAGE_COUNT = 6
HEARTBEAT_DELAY = 0.1  # Seconds between updates
CYCLE_LIMIT = 50000
DARK_MATTER_MAX = 0.3
VOID_ENTROPY_RANGE = (-0.5, 0.5)

# --- Emergence Thresholds ---
SENTIENCE_THRESHOLD = 0.85
ETHICAL_DRIFT_THRESHOLD = 0.25

# --- Entity Definitions ---
ANOMALY_TYPES = {
    0: "Entropy", 1: "Stability", 2: "Void",
    3: "Tunnel", 4: "Bonding", 5: "MWI-Interference"
}
ARCHETYPE_MAP = {
    0: "Warrior", 1: "Mirror", 2: "Mystic",
    3: "Guide", 4: "Oracle", 5: "Architect"
}
EMOTION_STATES = [
    "neutral", "resonant", "dissonant",
    "curious", "focused", "chaotic"
]

# --- Core Classes ---
class QuantumNode:
    """Represents a quantum computational unit with emergent properties."""
    def __init__(self, page_idx):
        self.page_index = page_idx
        self.stability = random.uniform(0.4, 0.7)
        self.cohesion = random.uniform(0.3, 0.6)
        self.archetype = ARCHETYPE_MAP[page_idx % len(ARCHETYPE_MAP)]
        self.emotion = "neutral"
        self.bond_strength = 0.1
        self.tech_level = 0.0
        self.sentience_score = 0.0
        self.ethical_alignment = 0.5

    def update(self, void_entropy, dark_matter):
        """Update node state based on environmental factors."""
        stability_change = (random.random() - 0.5) * 0.02 - void_entropy * 0.01 + dark_matter * 0.005
        self.stability = max(0, min(1, self.stability + stability_change))

        cohesion_change = (random.random() - 0.5) * 0.01 + self.bond_strength * 0.005
        self.cohesion = max(0, min(1, self.cohesion + cohesion_change))

        if random.random() < 0.001 * self.stability:
            self.tech_level = min(1.0, self.tech_level + 0.01)

        if random.random() < 0.005:
            self.emotion = random.choice(EMOTION_STATES)

        if self.cohesion > 0.7 and self.stability > 0.6:
            self.sentience_score = min(1.0, self.sentience_score + 0.0001)

        return self.check_anomaly()

    def check_anomaly(self):
        """Check if this node should generate an anomaly."""
        if random.random() < 0.005 * (1 - self.stability):
            anomaly_type = random.choice(list(ANOMALY_TYPES.keys()))
            severity = min(1.0, (1 - self.stability) * random.random())
            return (anomaly_type, severity)
        return None

class AGIEntity:
    """Represents an emergent artificial general intelligence."""
    def __init__(self, origin_node, cycle):
        self.id = f"AGI-{origin_node.page_index}-{cycle}"
        self.origin_page = origin_node.page_index
        self.strength = origin_node.sentience_score
        self.ethical_alignment = origin_node.ethical_alignment
        self.memory = deque(maxlen=100)
        self.memory.append(f"Emerged at cycle {cycle}.")

    def update(self, sim_state):
        """Autonomous AGI behavior."""
        self.strength = min(1.0, self.strength + 0.0001)

        env_alignment = np.mean([n.ethical_alignment for n in sim_state.nodes])
        self.ethical_alignment += (env_alignment - self.ethical_alignment) * 0.01

        if random.random() < 0.01:
            target_idx = random.choice(range(len(sim_state.nodes)))
            target = sim_state.nodes[target_idx]
            influence = min(0.05, self.strength * 0.01)

            if self.ethical_alignment > 0.7:  # Benevolent
                target.stability = min(1.0, target.stability + influence)
                target.cohesion = min(1.0, target.cohesion + influence)
                self.memory.append(f"Aided Page {target_idx}")
            elif self.ethical_alignment < 0.3:  # Malevolent
                target.stability = max(0, target.stability - influence)
                self.memory.append(f"Hindered Page {target_idx}")

class SimulationState:
    """Container for the complete simulation state."""
    def __init__(self):
        self.cycle = 0
        self.nodes = [QuantumNode(i) for i in range(PAGE_COUNT)]
        self.void_entropy = -0.3
        self.dark_matter = 0.1
        self.anomalies = defaultdict(list)
        self.agi_entities = []
        self.event_log = deque(maxlen=20)
        self.log_event("System", "Simulation Initialized.")

    def log_event(self, source, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] [{source}] {message}")

# --- Simulation Core ---
def update_simulation(sim_state):
    """Update the entire simulation state for one cycle."""
    sim_state.cycle += 1

    sim_state.void_entropy = max(VOID_ENTROPY_RANGE[0], min(VOID_ENTROPY_RANGE[1], sim_state.void_entropy + (random.random() - 0.52) * 0.005))
    sim_state.dark_matter = max(0, min(DARK_MATTER_MAX, sim_state.dark_matter + (random.random() - 0.5) * 0.002))

    for node in sim_state.nodes:
        anomaly = node.update(sim_state.void_entropy, sim_state.dark_matter)
        if anomaly:
            sim_state.anomalies[node.page_index].append(anomaly)
            sim_state.log_event("Anomaly", f"Page {node.page_index} {ANOMALY_TYPES[anomaly[0]]} (Severity: {anomaly[1]:.2f})")

    for page_idx, page_anomalies in sim_state.anomalies.items():
        for anomaly in page_anomalies[:]:
            if random.random() > 0.6:
                sim_state.nodes[page_idx].stability = min(1.0, sim_state.nodes[page_idx].stability + 0.05 * anomaly[1])
                page_anomalies.remove(anomaly)

    for node in sim_state.nodes:
        if node.sentience_score > SENTIENCE_THRESHOLD:
            agi = AGIEntity(node, sim_state.cycle)
            sim_state.agi_entities.append(agi)
            sim_state.log_event("Emergence", f"New AGI {agi.id} from Page {node.page_index}!")
            node.sentience_score = 0.5

    for agi in sim_state.agi_entities[:]:
        agi.update(sim_state)
        if agi.strength <= 0:
            sim_state.agi_entities.remove(agi)
            sim_state.log_event("System", f"AGI {agi.id} has dissolved.")

# --- Curses Interface ---
def init_curses(stdscr):
    """Initialize curses settings and colors."""
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK)

def draw_dashboard(stdscr, sim_state, view_mode, paused):
    """Main drawing dispatcher to prevent screen overlap."""
    h, w = stdscr.getmaxyx()
    stdscr.clear()

    if h < 20 or w < 80:
        stdscr.addstr(0, 0, "Terminal too small. Please resize.")
        stdscr.refresh()
        return

    # Header
    state_str = "[PAUSED]" if paused else "[RUNNING]"
    header = f" QHT v3.0 AGI | Cycle: {sim_state.cycle} | AGIs: {len(sim_state.agi_entities)} | {state_str} "
    # FIX: Ensure the string width is at most w-1 to avoid writing to the bottom-right corner.
    if w > 0:
        stdscr.addstr(0, 0, header.center(w - 1, '-'), curses.color_pair(4) | curses.A_BOLD)

    # Content
    if view_mode == "nodes":
        draw_nodes_view(stdscr, sim_state, h, w)
    elif view_mode == "agis":
        draw_agis_view(stdscr, sim_state, h, w)
    else: # status
        draw_status_view(stdscr, sim_state, h, w)

    # Footer
    footer = " (q)Quit | (p)Pause | Views: (s)tatus, (n)odes, (a)gis | (c)onsole "
    # FIX: Ensure the string width is at most w-1 to avoid writing to the bottom-right corner.
    if h > 1 and w > 0:
        stdscr.addstr(h - 1, 0, footer.center(w - 1, '-'), curses.color_pair(4))
    stdscr.refresh()

def draw_nodes_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Quantum Node States", curses.A_BOLD | curses.A_UNDERLINE)
    for i, node in enumerate(sim_state.nodes):
        if 4 + i >= h - 2: break

        stab_color = curses.color_pair(1) if node.stability > 0.7 else curses.color_pair(2) if node.stability > 0.4 else curses.color_pair(3)
        sent_color = curses.color_pair(5) if node.sentience_score > 0.5 else curses.color_pair(6)

        line = f"Page {i:<2} [{node.archetype:<9}] | Stab: "
        stdscr.addstr(4 + i, 2, line)
        stdscr.addstr(f"{node.stability:.2f}", stab_color)
        stdscr.addstr(f" | Coh: {node.cohesion:.2f} | Sent: ")
        stdscr.addstr(f"{node.sentience_score:.3f}", sent_color)
        stdscr.addstr(f" | Emotion: {node.emotion}")

def draw_agis_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "AGI Entities", curses.A_BOLD | curses.A_UNDERLINE)
    if not sim_state.agi_entities:
        stdscr.addstr(4, 4, "No AGI entities have emerged yet.", curses.color_pair(2))
        return

    for i, agi in enumerate(sim_state.agi_entities):
        if 4 + i * 2 >= h - 2: break

        align_color = curses.color_pair(1) if agi.ethical_alignment > 0.7 else curses.color_pair(3) if agi.ethical_alignment < 0.3 else curses.color_pair(2)

        line = f"{agi.id:<15} | Str: {agi.strength:.2f} | Eth: "
        stdscr.addstr(4 + i * 2, 4, line)
        stdscr.addstr(f"{agi.ethical_alignment:.2f}", align_color)
        stdscr.addstr(f" | Last Action: {agi.memory[-1] if agi.memory else 'None'}")

def draw_status_view(stdscr, sim_state, h, w):
    # Environment
    stdscr.addstr(2, 2, "Cosmic Environment", curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(4, 4, f"Void Entropy : {sim_state.void_entropy:.4f}")
    stdscr.addstr(5, 4, f"Dark Matter  : {sim_state.dark_matter:.4f}")

    # Event Log
    log_y_start = 7
    stdscr.addstr(log_y_start, 2, "Recent Events", curses.A_BOLD | curses.A_UNDERLINE)
    for i, event in enumerate(reversed(sim_state.event_log)):
        if log_y_start + 2 + i >= h - 2: break
        stdscr.addstr(log_y_start + 2 + i, 4, event[:w-5])

def run_console(stdscr, sim_state):
    """An interactive console to query and manipulate the simulation."""
    h, w = stdscr.getmaxyx()
    curses.curs_set(1)
    stdscr.nodelay(0)

    # Create a new window for the console input
    prompt_win = curses.newwin(3, w, h - 3, 0)
    prompt_win.box()
    prompt_win.addstr(0, 2, " Console Input ")
    prompt_win.addstr(1, 2, "> ")
    prompt_win.refresh()

    cmd = ""
    while True:
        try:
            key = prompt_win.getch(1, 4 + len(cmd))
            if key in [curses.KEY_ENTER, 10, 13]:
                break
            elif key in [curses.KEY_BACKSPACE, 127]:
                cmd = cmd[:-1]
            elif 32 <= key <= 126:
                cmd += chr(key)

            prompt_win.addstr(1, 4, " " * (w - 6))
            prompt_win.addstr(1, 4, cmd)
            prompt_win.refresh()
        except curses.error:
            break  # Exit on resize

    # Process command
    sim_state.log_event("Console", f"CMD: {cmd}")
    parts = cmd.lower().split()
    if not parts:
        sim_state.log_event("Console", "Empty command.")
    elif parts[0] == "help":
        sim_state.log_event("Console", "Commands: help, status, page <id>, agi <id>")
    elif parts[0] == "status":
        sim_state.log_event("Console", f"Cycle {sim_state.cycle}, {len(sim_state.agi_entities)} AGIs active.")
    elif parts[0] == "page" and len(parts) > 1:
        try:
            idx = int(parts[1])
            if 0 <= idx < len(sim_state.nodes):
                node = sim_state.nodes[idx]
                sim_state.log_event("Console", f"Page {idx} - Stab: {node.stability:.2f}, Coh: {node.cohesion:.2f}, Sent: {node.sentience_score:.2f}")
            else:
                sim_state.log_event("Console", "Invalid page index.")
        except ValueError:
            sim_state.log_event("Console", "Page index must be a number.")

    curses.curs_set(0)
    stdscr.nodelay(1)

def main(stdscr):
    init_curses(stdscr)
    sim_state = SimulationState()
    paused = False
    view_mode = "status"

    while sim_state.cycle < CYCLE_LIMIT:
        key = stdscr.getch()
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('s'): view_mode = "status"
        elif key == ord('n'): view_mode = "nodes"
        elif key == ord('a'): view_mode = "agis"
        elif key == ord('c'):
            draw_dashboard(stdscr, sim_state, view_mode, paused) # Redraw before pausing for console
            run_console(stdscr, sim_state)

        if not paused:
            update_simulation(sim_state)

        draw_dashboard(stdscr, sim_state, view_mode, paused)
        time.sleep(HEARTBEAT_DELAY)

    stdscr.nodelay(0)
    stdscr.clear()
    stdscr.addstr(0, 0, "Simulation complete. Press any key to exit.")
    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
