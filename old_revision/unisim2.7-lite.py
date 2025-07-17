import math
import random
import time
import datetime
from collections import defaultdict, deque
import curses

# --- Lite Version Constants ---
OCTREE_DEPTH = 3
PAGE_COUNT = 4
HEARTBEAT = 10
CYCLE_LIMIT = 100000
SIGIL_LEN = 80
ANOMALY_TYPES = {0: "Entropy", 1: "Stability", 2: "Void", 3: "Tunnel", 4: "Bonding"}
ARCHETYPE_MAP = {0: "Warrior", 1: "Mirror", 2: "Mystic", 3: "Guide", 4: "Oracle", 5: "Architect", 6: "Warden"}

# Simplified Classes
class Qubit352:
    def __init__(self):
        self.alpha = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        self.beta = complex(random.uniform(-1, 1), random.uniform(-1, 1))
        self.coherence_time = random.uniform(0.5, 1.0)

    def measure(self):
        return random.random() < abs(self.alpha)**2

class OctNode:
    def __init__(self, depth, page_index):
        self.st = Qubit352()
        self.page_index = page_index
        self.stabilityPct = 0.0
        self.social_cohesion = 0.0
        self.archetype = ARCHETYPE_MAP[page_index % len(ARCHETYPE_MAP)]
        self.emotion = "neutral"
        self.bond_strength = 0.0

class Anomaly:
    def __init__(self, cycle, page_idx, anomaly_type, severity):
        self.cycle = cycle
        self.page_idx = page_idx
        self.anomaly_type = anomaly_type
        self.severity = severity

def init_simulation():
    global roots, anomalies_per_page
    roots = []
    anomalies_per_page = defaultdict(list)
    for p_idx in range(PAGE_COUNT):
        root_node = OctNode(OCTREE_DEPTH, p_idx)
        roots.append(root_node)

def update_node_dynamics(node, p_idx):
    global voidEntropy

    # Simplified dynamics
    node.stabilityPct = max(0, min(1, node.stabilityPct + (random.random() - 0.5) * 0.01))
    node.social_cohesion = max(0, min(1, node.social_cohesion + (random.random() - 0.5) * 0.005))
    node.bond_strength = max(0, node.bond_strength - 0.003 + random.random() * 0.002)

    # Anomaly triggering
    if random.random() < 0.005 * (1 - node.stabilityPct):
        anomaly_type = random.choice(list(ANOMALY_TYPES.keys()))
        severity = min(1.0, (1 - node.stabilityPct) * random.random())
        trigger_anomaly(anomaly_type, p_idx, severity)

def trigger_anomaly(anomaly_type, page_idx, severity):
    global total_anomalies_triggered, anomaly_type_counts, anomalies_per_page
    anomalies_per_page[page_idx].append(Anomaly(cycle_num, page_idx, anomaly_type, severity))
    total_anomalies_triggered += 1
    anomaly_type_counts[anomaly_type] += 1

def handle_anomalies():
    global total_anomalies_fixed, anomalies_per_page
    for page_idx, anomalies in anomalies_per_page.items():
        for anomaly in anomalies[:]:
            if random.random() > 0.7:  # 70% fix rate
                roots[page_idx].stabilityPct = min(1.0, roots[page_idx].stabilityPct + 0.05)
                anomalies.remove(anomaly)
                total_anomalies_fixed += 1

def evolve_archetypes():
    for node in roots:
        if random.random() < 0.001 and node.stabilityPct > 0.7:
            node.archetype = random.choice(list(ARCHETYPE_MAP.values()))
            node.emotion = random.choice(["calm", "focused", "energized"])

def update_void_entropy():
    global voidEntropy
    voidEntropy = max(-1.0, min(0.5, voidEntropy + (random.random() - 0.52) * 0.005))

# Terminal Dashboard
def init_dashboard(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

def draw_dashboard(stdscr, cycle_num, voidEntropy, total_anomalies_triggered, total_anomalies_fixed, anomaly_type_counts, roots):
    # Clear screen
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    # Header
    stdscr.addstr(0, 0, f"QuantumHeapTranscendence Lite | Cycle: {cycle_num}/{CYCLE_LIMIT}")
    stdscr.addstr(1, 0, "-" * min(width, 80))  # Limit to 80 chars to avoid curses error

    # Global metrics
    stdscr.addstr(3, 0, "Global Metrics:")
    stdscr.addstr(4, 2, f"Void Entropy: {voidEntropy:.4f}")
    stdscr.addstr(5, 2, f"Anomalies: {total_anomalies_triggered} triggered, {total_anomalies_fixed} fixed")

    # Page states
    stdscr.addstr(7, 0, "Page States:")
    for i, node in enumerate(roots):
        if i < 10 and 8 + i < height - 3:  # Only show first 10 pages and prevent overflow
            stdscr.addstr(8 + i, 2,
                f"Page {i}: {node.archetype[:8]:<8} | "
                f"Stab: {node.stabilityPct:.2f} | "
                f"Cohesion: {node.social_cohesion:.2f} | "
                f"Bond: {node.bond_strength:.2f} | "
                f"Anoms: {len(anomalies_per_page[i])}"
            )

    # Anomaly distribution
    if 20 < height - 3:
        stdscr.addstr(20, 0, "Anomaly Distribution:")
        for i, (atype, count) in enumerate(anomaly_type_counts.items()):
            if 21 + i < height - 3:
                stdscr.addstr(21 + i, 2, f"{ANOMALY_TYPES[atype]}: {count}")

    # Footer
    if height - 2 > 0:
        stdscr.addstr(height - 2, 0, "-" * min(width, 80))
    if height - 1 > 0:
        stdscr.addstr(height - 1, 0, "Press 'q' to quit | 'p' to pause")

    stdscr.refresh()

def main(stdscr):
    # Initialize global variables
    global cycle_num, voidEntropy, total_anomalies_triggered, total_anomalies_fixed
    global anomaly_type_counts, roots, anomalies_per_page

    cycle_num = 0
    voidEntropy = -0.3
    total_anomalies_triggered = 0
    total_anomalies_fixed = 0
    anomaly_type_counts = defaultdict(int)

    init_dashboard(stdscr)
    init_simulation()

    paused = False
    running = True
    last_update = 0

    while running and cycle_num < CYCLE_LIMIT:
        current_time = time.time()

        # Handle input
        try:
            key = stdscr.getch()
            if key == ord('q'):
                running = False
            elif key == ord('p'):
                paused = not paused
        except:
            pass

        if not paused:
            # Update simulation
            cycle_num += 1
            if cycle_num % HEARTBEAT == 0:
                for i, node in enumerate(roots):
                    update_node_dynamics(node, i)
                handle_anomalies()
                evolve_archetypes()
                update_void_entropy()

            # Update dashboard
            if current_time - last_update > 0.1:
                try:
                    draw_dashboard(stdscr, cycle_num, voidEntropy, total_anomalies_triggered,
                                 total_anomalies_fixed, anomaly_type_counts, roots)
                except curses.error:
                    pass  # Ignore screen drawing errors when terminal is resized
                last_update = current_time

    # Final message
    try:
        stdscr.clear()
        stdscr.addstr(0, 0, "Simulation completed!")
        stdscr.addstr(1, 0, f"Final cycle: {cycle_num}")
        stdscr.addstr(2, 0, f"Total anomalies: {total_anomalies_triggered} triggered, {total_anomalies_fixed} fixed")
        stdscr.addstr(4, 0, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()
    except curses.error:
        pass

if __name__ == "__main__":
    curses.wrapper(main)
