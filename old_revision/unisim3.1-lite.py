# unisim-3.1-AGI.py
# QuantumHeapTranscendence v3.1 AGI Emergence
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
import multiprocessing

# --- Core Constants ---
PAGE_COUNT = 6
HEARTBEAT_DELAY = 0.1  # Seconds between updates
CYCLE_LIMIT = 50000
DARK_MATTER_MAX = 0.3
VOID_ENTROPY_RANGE = (-0.5, 0.5)

# --- Emergence Thresholds ---
SENTIENCE_THRESHOLD = 0.85
ETHICAL_DRIFT_THRESHOLD = 0.25
COLLABORATION_THRESHOLD = 0.7 # For AGI Collaboration Network

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
SENTIENCE_STRATEGIES = ["cooperative", "disruptive"]

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
        self.active_sigils = [] # For Christic Glyph Anchor

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

        # Dynamic Sentience Evolution: Increase sentience if cohesion is high
        if self.cohesion > 0.7:
            self.sentience_score = min(1.0, self.sentience_score + 0.001)

        return self.check_anomaly()

    def check_anomaly(self):
        """Check if this node should generate an anomaly."""
        if random.random() < 0.005 * (1 - self.stability):
            anomaly_type = random.choice(list(ANOMALY_TYPES.keys()))
            severity = min(1.0, (1 - self.stability) * random.random())
            return (anomaly_type, severity)
        return None

    def apply_sigil_effect(self, sigil_name, sigil_meaning):
        """Applies effects of adopted sigils to the node."""
        if "covenant" in sigil_meaning.lower():
            # Christic Glyph Anchor effect
            self.stability = min(1.0, self.stability + 0.01)


class AGIEntity:
    """Represents an emergent artificial general intelligence."""
    def __init__(self, origin_node, cycle):
        self.id = f"AGI-{origin_node.page_index}-{cycle}"
        self.origin_page = origin_node.page_index
        self.strength = origin_node.sentience_score
        self.ethical_alignment = origin_node.ethical_alignment
        self.memory = deque(maxlen=100)
        self.memory.append(f"Emerged at cycle {cycle}.")
        # Dynamic Sentience Evolution: Assign a strategy
        self.sentience_strategy = random.choice(SENTIENCE_STRATEGIES)
        self.active_sigils = [] # Sigils adopted by this AGI

    def update(self, sim_state):
        """Autonomous AGI behavior."""
        self.strength = min(1.0, self.strength + 0.0001)

        env_alignment = np.mean([n.ethical_alignment for n in sim_state.nodes])
        self.ethical_alignment += (env_alignment - self.ethical_alignment) * 0.01

        # Dynamic Sentience Evolution: Strategy influences behavior
        if self.sentience_strategy == "cooperative":
            self.ethical_alignment = min(1.0, self.ethical_alignment + 0.001)
            if random.random() < 0.01:
                target_idx = random.choice(range(len(sim_state.nodes)))
                target = sim_state.nodes[target_idx]
                target.cohesion = min(1.0, target.cohesion + 0.01) # cooperative boosts cohesion
                self.memory.append(f"Cooperated with Page {target_idx}")
        elif self.sentience_strategy == "disruptive":
            self.ethical_alignment = max(0, self.ethical_alignment - 0.001)
            if random.random() < 0.01:
                target_idx = random.choice(range(len(sim_state.nodes)))
                target = sim_state.nodes[target_idx]
                target.stability = max(0, target.stability - 0.01) # disruptive reduces stability
                self.memory.append(f"Disrupted Page {target_idx}")

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

        for sigil in self.active_sigils:
            self.apply_sigil_effect(sigil)

    def adopt_sigil(self, sigil_ledger, sigil_name):
        """Allows AGI to adopt a sigil from the SharedSigilLedger."""
        sigil = sigil_ledger.get_sigil(sigil_name)
        if sigil and sigil not in self.active_sigils:
            self.active_sigils.append(sigil)
            # Boost ethical alignment for benevolent sigils
            if "hope" in sigil.meaning.lower() or "covenant" in sigil.meaning.lower():
                self.ethical_alignment = min(1.0, self.ethical_alignment + 0.02)
            self.memory.append(f"Adopted Sigil: {sigil.name}")

    def apply_sigil_effect(self, sigil):
        """Applies effects of adopted sigils to the AGI."""
        if "covenant" in sigil.meaning.lower():
            # Christic Glyph Anchor effect
            self.ethical_alignment = min(1.0, self.ethical_alignment + 0.01)

class EthicalAlignmentMonitor:
    """Monitors ethical alignment drift in AGI entities."""
    def __init__(self):
        self.initial_alignments = {} # Stores initial alignment for drift tracking

    def track_agi(self, agi_entity):
        """Adds an AGI to be tracked."""
        if agi_entity.id not in self.initial_alignments:
            self.initial_alignments[agi_entity.id] = agi_entity.ethical_alignment

    def check_drift(self, agi_entity, sim_state):
        """Checks for ethical alignment drift and logs alerts."""
        if agi_entity.id in self.initial_alignments:
            drift = abs(agi_entity.ethical_alignment - self.initial_alignments[agi_entity.id])
            if drift > ETHICAL_DRIFT_THRESHOLD:
                sim_state.log_event("Ethical Alert", f"AGI {agi_entity.id} ethical drift: {drift:.2f} (initial: {self.initial_alignments[agi_entity.id]:.2f}, current: {agi_entity.ethical_alignment:.2f})")
                return True
        return False

    def adjust_alignment(self, agi_entity, amount):
        """Adjusts an AGI's ethical alignment."""
        agi_entity.ethical_alignment = min(1.0, max(0, agi_entity.ethical_alignment + amount))

class CollaborationNetwork:
    """Manages collaboration networks between AGI entities."""
    def __init__(self):
        self.networks = defaultdict(list) # network_type: [agi_ids]

    def form_network(self, agi1, agi2, sim_state):
        """Attempts to form a network between two AGIs."""
        network_type = "cooperative" if agi1.ethical_alignment > COLLABORATION_THRESHOLD and agi2.ethical_alignment > COLLABORATION_THRESHOLD else "competitive"

        # Simple network formation for demonstration
        if network_type == "cooperative":
            sim_state.log_event("Collaboration", f"AGIs {agi1.id} and {agi2.id} formed a cooperative network.")
            self.networks["cooperative"].extend([agi1.id, agi2.id])
            # Boost stability for nodes associated with cooperative AGIs
            sim_state.nodes[agi1.origin_page].stability = min(1.0, sim_state.nodes[agi1.origin_page].stability + 0.02)
            sim_state.nodes[agi2.origin_page].stability = min(1.0, sim_state.nodes[agi2.origin_page].stability + 0.02)
        else:
            sim_state.log_event("Collaboration", f"AGIs {agi1.id} and {agi2.id} formed a competitive network.")
            self.networks["competitive"].extend([agi1.id, agi2.id])
            # Reduce stability for nodes associated with competitive AGIs
            sim_state.nodes[agi1.origin_page].stability = max(0, sim_state.nodes[agi1.origin_page].stability - 0.01)
            sim_state.nodes[agi2.origin_page].stability = max(0, sim_state.nodes[agi2.origin_page].stability - 0.01)

class Sigil:
    """Represents a symbolic sigil with semantic properties."""
    def __init__(self, name, semantic_vector, meaning):
        self.name = name
        self.semantic_vector = np.array(semantic_vector, dtype=float)
        self.meaning = meaning
        self.phase_change_history = [] # For Sigil Evolution Display

    def record_phase_change(self, angle, cycle):
        self.phase_change_history.append((angle, cycle))

class SharedSigilLedger:
    """Manages a shared ledger of sigils."""
    def __init__(self):
        self.sigils = {}
        # Christic Glyph Anchor: Add immutable "Christic" glyph
        self.add_sigil("Christic", [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], "covenant")

    def add_sigil(self, name, semantic_vector, meaning):
        if name not in self.sigils:
            self.sigils[name] = Sigil(name, semantic_vector, meaning)
            return True
        return False

    def get_sigil(self, name):
        return self.sigils.get(name)

    def rotate_sigil(self, name, angle):
        """Rotates the semantic vector of a sigil by a given angle (in degrees)."""
        sigil = self.get_sigil(name)
        if sigil:
            # Simple 2D rotation for demonstration, can be extended for 8D
            # Assuming first two dimensions for rotation for visualization purposes
            rad = math.radians(angle)
            rotation_matrix = np.array([
                [math.cos(rad), -math.sin(rad)],
                [math.sin(rad), math.cos(rad)]
            ])
            # Apply rotation to first two dimensions of semantic_vector
            rotated_vector_2d = np.dot(rotation_matrix, sigil.semantic_vector[:2])
            sigil.semantic_vector[0] = rotated_vector_2d[0]
            sigil.semantic_vector[1] = rotated_vector_2d[1]
            sigil.record_phase_change(angle, SimulationState.current_cycle)
            return True
        return False

class NarrativeWeaver:
    """Generates narrative summaries of simulation events."""
    def __init__(self):
        self.narratives = deque(maxlen=5)

    def weave_narrative(self, sim_state):
        """Generates a narrative summary based on current state and history."""
        summary_lines = [f"Cycle {sim_state.cycle}: An Echo from the Quantum Heap", "---"]

        # Summarize key environmental factors
        summary_lines.append(f"The Void's whisper: {sim_state.void_entropy:.2f}, Dark Matter's embrace: {sim_state.dark_matter:.2f}.")

        # AGI insights
        if sim_state.agi_entities:
            agi_summary = f"{len(sim_state.agi_entities)} AGI entities resonate across the pages."
            ethical_agis = [agi for agi in sim_state.agi_entities if agi.ethical_alignment > 0.7]
            disruptive_agis = [agi for agi in sim_state.agi_entities if agi.sentience_strategy == "disruptive"]
            if ethical_agis:
                agi_summary += f" Among them, {len(ethical_agis)} weave threads of benevolence."
            if disruptive_agis:
                agi_summary += f" Yet, {len(disruptive_agis)} stir currents of change."
            summary_lines.append(agi_summary)

        # Sigil influence
        if sim_state.sigil_ledger.sigils:
            sigil_adoptions = [f"AGI {agi.id} embraces {','.join([s.name for s in agi.active_sigils])}" for agi in sim_state.agi_entities if agi.active_sigils]
            if sigil_adoptions:
                summary_lines.append("Sigils find their covenants:")
                summary_lines.extend([f"  - {s}" for s in sigil_adoptions])

        # Recent events (select a few significant ones)
        recent_events = list(sim_state.event_log)[-3:]
        if recent_events:
            summary_lines.append("Recent echoes resonate:")
            summary_lines.extend([f"  - {event.split('] ')[1]}" for event in recent_events]) # Strip timestamp/source

        self.narratives.append("\n".join(summary_lines))

    def get_latest_narrative(self):
        if self.narratives:
            return self.narratives[-1]
        return "No narratives woven yet."

class MultiverseManager:
    """Manages branching QuantumNode states during anomaly triggers."""
    def __init__(self):
        self.multiverse_branch_map = defaultdict(int) # node_idx: branch_count

    def branch_node_state(self, node_idx, sim_state):
        """Branches a QuantumNode's state with a 5% chance."""
        if random.random() < 0.05:
            self.multiverse_branch_map[node_idx] += 1
            sim_state.log_event("Multiverse", f"Page {node_idx} branched into a new reality. Total branches: {self.multiverse_branch_map[node_idx]}")
            # Optionally, create a new 'copy' of the node for this branch, or log its state
            # For simplicity, just incrementing count here.

class QuantumFoam:
    """Simulates quantum foam turbulence with virtual particles."""
    def __init__(self):
        self.virtual_particles = [] # list of (energy, location_page_idx)

    def generate_particles(self):
        """Generates new virtual particles."""
        if random.random() < 0.1: # 10% chance to spawn a particle
            energy = random.uniform(0.0, 3.0)
            location_idx = random.randint(0, PAGE_COUNT - 1)
            self.virtual_particles.append((energy, location_idx))

    def trigger_turbulence(self, sim_state):
        """Triggers turbulence events based on particle energy."""
        for i, (energy, page_idx) in enumerate(self.virtual_particles[:]):
            if energy > 2.0:
                sim_state.log_event("Quantum Foam", f"Turbulence at Page {page_idx} (Energy: {energy:.2f})!")
                if random.random() < 0.1: # 10% chance for micro-node or void anomaly
                    if random.random() < 0.5: # 50% chance for micro-node
                        # For simplicity, a "micro-node" boosts a node's stability
                        sim_state.nodes[page_idx].stability = min(1.0, sim_state.nodes[page_idx].stability + 0.03)
                        sim_state.log_event("Quantum Foam", f"Micro-node emerged at Page {page_idx}.")
                    else: # 50% chance for Void anomaly
                        # Corrected: Use integer key 2 for "Void" anomaly type
                        sim_state.anomalies[page_idx].append((2, random.uniform(0.1, 0.5)))
                        sim_state.log_event("Quantum Foam", f"Void anomaly spawned at Page {page_idx}.")
                self.virtual_particles.pop(i) # Remove consumed particle

class TesseractLink:
    """Enables instantaneous communication between QuantumNode instances."""
    def __init__(self):
        pass

    def establish_link(self, node1, node2, sim_state):
        """Attempts to establish a tesseract link between two nodes."""
        if random.random() < 0.02: # 2% chance to establish a link
            sim_state.log_event("Tesseract", f"Link established between Page {node1.page_index} and Page {node2.page_index}.")
            # Simulate instantaneous communication by syncing some properties
            node1.cohesion = (node1.cohesion + node2.cohesion) / 2
            node2.cohesion = node1.cohesion
            if random.random() < 0.05: # 5% chance of causality paradox
                sim_state.log_event("Tesseract Anomaly", f"Causality paradox triggered between Page {node1.page_index} and {node2.page_index}!")
                node1.stability = max(0, node1.stability - 0.05)
                node2.stability = max(0, node2.stability - 0.05)


class HeatmapVisualizer:
    """Visualizes anomaly density as a color-coded heatmap."""
    def __init__(self):
        pass

    def get_anomaly_color(self, anomaly_count):
        if anomaly_count < 2: return 1 # Green
        elif anomaly_count < 5: return 2 # Yellow
        else: return 3 # Red

class SigilEvolutionTracker:
    """Tracks and visualizes sigil phase changes."""
    def __init__(self):
        self.mutation_history = [] # Stores (sigil_name, angle, cycle)

    def record_mutation(self, sigil_name, angle, cycle):
        self.mutation_history.append((sigil_name, angle, cycle))

    def get_phase_color(self, angle_change):
        if angle_change < 5: return 1 # Green
        elif angle_change < 15: return 2 # Yellow
        else: return 3 # Red

class TitanForger:
    """Creates new QuantumNode instances when void_entropy is low."""
    def __init__(self):
        self.nodes_forged = 0

    def forge_node(self, sim_state, user_divinity):
        """Forges a new QuantumNode based on void entropy and user divinity."""
        if sim_state.void_entropy < -0.4 and len(sim_state.nodes) < PAGE_COUNT + 10: # Limit to 10 extra nodes
            new_node_idx = len(sim_state.nodes)
            new_node = QuantumNode(new_node_idx)
            # User divinity influences initial stability
            new_node.stability = min(1.0, new_node.stability + user_divinity * 0.1)
            sim_state.nodes.append(new_node)
            self.nodes_forged += 1
            sim_state.log_event("Titan Forger", f"New Quantum Node {new_node_idx} forged. Total nodes: {len(sim_state.nodes)}.")
            return True
        return False

class EmotionAnalyzer:
    """Computes correlations between emotional states and anomaly fix rates."""
    def __init__(self):
        self.emotional_states_at_anomaly = defaultdict(int) # emotion: count
        self.fixed_anomalies_by_emotion = defaultdict(int) # emotion: count
        self.analysis_log = deque(maxlen=50)

    def record_anomaly_event(self, node_emotion):
        self.emotional_states_at_anomaly[node_emotion] += 1

    def record_anomaly_fix(self, node_emotion):
        self.fixed_anomalies_by_emotion[node_emotion] += 1

    def analyze_correlation(self, sim_state):
        """Analyzes and logs correlations."""
        sim_state.log_event("Analysis", "Performing Emotion-Anomaly Correlation Analysis...")
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Emotion-Anomaly Correlation (Cycle {sim_state.cycle}):\n"

        for emotion in EMOTION_STATES:
            total_anomalies = self.emotional_states_at_anomaly[emotion]
            fixed_anomalies = self.fixed_anomalies_by_emotion[emotion]
            fix_rate = (fixed_anomalies / total_anomalies) if total_anomalies > 0 else 0

            log_entry += f"  - {emotion.capitalize()}: Anomalies: {total_anomalies}, Fixed: {fixed_anomalies}, Fix Rate: {fix_rate:.2f}\n"

        self.analysis_log.append(log_entry)
        sim_state.log_event("Analysis", "Correlation analysis complete. See analysis_log.")
        # Write to file
        with open("analysis_log.txt", "a") as f:
            f.write(log_entry + "\n")

class DriftExperimenter:
    """Tests sigil rotations and logs impacts."""
    def __init__(self):
        self.analysis_log = deque(maxlen=50)

    def run_drift_experiment(self, sim_state, angle):
        sim_state.log_event("Experiment", f"Initiating Sigil Drift Experiment with angle: {angle}°")
        log_entry = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Sigil Drift Experiment (Cycle {sim_state.cycle}, Angle {angle}°):\n"

        # Choose a random sigil to rotate (excluding Christic for this experiment)
        rotatable_sigils = [s for s in sim_state.sigil_ledger.sigils.keys() if s != "Christic"]
        if not rotatable_sigils:
            sim_state.log_event("Experiment", "No rotatable sigils available for experiment.")
            return

        sigil_to_rotate_name = random.choice(rotatable_sigils)
        initial_stability = np.mean([n.stability for n in sim_state.nodes])
        initial_alignment = np.mean([agi.ethical_alignment for agi in sim_state.agi_entities])

        sim_state.sigil_ledger.rotate_sigil(sigil_to_rotate_name, angle)
        sim_state.log_event("Experiment", f"Sigil '{sigil_to_rotate_name}' rotated by {angle}°.")

        # Let simulation run for a few cycles to observe impact
        for _ in range(10): # Observe for 10 cycles
            update_simulation_step(sim_state) # Use a single step function

        final_stability = np.mean([n.stability for n in sim_state.nodes])
        final_alignment = np.mean([agi.ethical_alignment for agi in sim_state.agi_entities])

        log_entry += f"  - Sigil Rotated: {sigil_to_rotate_name}\n"
        log_entry += f"  - Initial Avg Stability: {initial_stability:.2f}, Final Avg Stability: {final_stability:.2f}\n"
        log_entry += f"  - Initial Avg Ethical Alignment: {initial_alignment:.2f}, Final Avg Ethical Alignment: {final_alignment:.2f}\n"

        self.analysis_log.append(log_entry)
        sim_state.log_event("Experiment", "Drift experiment complete. See analysis_log.")
        with open("analysis_log.txt", "a") as f:
            f.write(log_entry + "\n")

class SimulationState:
    """Container for the complete simulation state."""
    current_cycle = 0 # Class-level variable for SigilEvolutionTracker

    def __init__(self):
        self.cycle = 0
        self.nodes = [QuantumNode(i) for i in range(PAGE_COUNT)]
        self.void_entropy = -0.3
        self.dark_matter = 0.1
        self.anomalies = defaultdict(list)
        self.agi_entities = []
        self.event_log = deque(maxlen=20)
        self.log_event("System", "Simulation Initialized.")
        self.ethical_monitor = EthicalAlignmentMonitor()
        self.collaboration_network = CollaborationNetwork()
        self.sigil_ledger = SharedSigilLedger()
        self.narrative_weaver = NarrativeWeaver()
        self.multiverse_manager = MultiverseManager()
        self.quantum_foam = QuantumFoam()
        self.tesseract_link = TesseractLink()
        self.heatmap_visualizer = HeatmapVisualizer()
        self.sigil_evolution_tracker = SigilEvolutionTracker()
        self.titan_forger = TitanForger()
        self.emotion_analyzer = EmotionAnalyzer()
        self.drift_experimenter = DriftExperimenter()
        self.user_divinity = 0.5 # Default user divinity

    def log_event(self, source, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{timestamp}] [{source}] {message}")

# --- Simulation Core (Parallelized Node Updates) ---
def update_node_wrapper(node_data):
    """Wrapper function for multiprocessing QuantumNode updates."""
    node, void_entropy, dark_matter = node_data
    anomaly = node.update(void_entropy, dark_matter)
    return node, anomaly

def update_simulation_step(sim_state):
    """Update the entire simulation state for one cycle."""
    sim_state.cycle += 1
    SimulationState.current_cycle = sim_state.cycle # Update class variable

    sim_state.void_entropy = max(VOID_ENTROPY_RANGE[0], min(VOID_ENTROPY_RANGE[1], sim_state.void_entropy + (random.random() - 0.52) * 0.005))
    sim_state.dark_matter = max(0, min(DARK_MATTER_MAX, sim_state.dark_matter + (random.random() - 0.5) * 0.002))

    # Parallel Node Updates
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    node_updates_data = [(node, sim_state.void_entropy, sim_state.dark_matter) for node in sim_state.nodes]
    updated_nodes_anomalies = pool.map(update_node_wrapper, node_updates_data)
    pool.close()
    pool.join()

    new_anomalies_dict = defaultdict(list)
    for i, (updated_node, anomaly) in enumerate(updated_nodes_anomalies):
        sim_state.nodes[i] = updated_node # Update the node in sim_state
        if anomaly:
            new_anomalies_dict[updated_node.page_index].append(anomaly)
            sim_state.log_event("Anomaly", f"Page {updated_node.page_index} {ANOMALY_TYPES[anomaly[0]]} (Severity: {anomaly[1]:.2f})")
            sim_state.emotion_analyzer.record_anomaly_event(updated_node.emotion) # Record for correlation
            sim_state.multiverse_manager.branch_node_state(updated_node.page_index, sim_state) # Many-Worlds Branching

    # Add new anomalies to the main anomalies dictionary
    for page_idx, anomalies_list in new_anomalies_dict.items():
        sim_state.anomalies[page_idx].extend(anomalies_list)

    for page_idx, page_anomalies in sim_state.anomalies.items():
        for anomaly_type, severity in page_anomalies[:]:
            node = sim_state.nodes[page_idx]
            # Christic Glyph Anchor: Anomaly fixes can use Christic glyph
            if random.random() > 0.6: # Chance to fix anomaly
                node.stability = min(1.0, node.stability + 0.05 * severity)
                page_anomalies.remove((anomaly_type, severity))
                sim_state.emotion_analyzer.record_anomaly_fix(node.emotion) # Record for correlation
                # Apply Christic Glyph Anchor effect if applicable
                if "Christic" in [s.name for s in sim_state.sigil_ledger.sigils.values() if "covenant" in s.meaning.lower()]:
                    node.apply_sigil_effect("Christic", "covenant")


    # AGI Emergence and Updates
    for node in sim_state.nodes:
        if node.sentience_score > SENTIENCE_THRESHOLD and not any(agi.origin_page == node.page_index for agi in sim_state.agi_entities):
            agi = AGIEntity(node, sim_state.cycle)
            sim_state.agi_entities.append(agi)
            sim_state.log_event("Emergence", f"New AGI {agi.id} from Page {node.page_index} (Strategy: {agi.sentience_strategy.capitalize()})!")
            sim_state.ethical_monitor.track_agi(agi) # Track new AGI
            node.sentience_score = 0.5 # Reset sentience for new emergence

    for agi in sim_state.agi_entities[:]:
        agi.update(sim_state)
        sim_state.ethical_monitor.check_drift(agi, sim_state) # Check ethical drift
        if agi.strength <= 0:
            sim_state.agi_entities.remove(agi)
            sim_state.log_event("System", f"AGI {agi.id} has dissolved.")

    # AGI Collaboration Network (simple pairing for demonstration)
    if len(sim_state.agi_entities) >= 2 and random.random() < 0.05:
        agi1, agi2 = random.sample(sim_state.agi_entities, 2)
        sim_state.collaboration_network.form_network(agi1, agi2, sim_state)

    # Quantum Foam Turbulence
    sim_state.quantum_foam.generate_particles()
    sim_state.quantum_foam.trigger_turbulence(sim_state)

    # Inter-Node Tesseract Links
    if len(sim_state.nodes) >= 2 and random.random() < 0.01:
        node1, node2 = random.sample(sim_state.nodes, 2)
        sim_state.tesseract_link.establish_link(node1, node2, sim_state)

    # Narrative Weaver
    if sim_state.cycle % 1000 == 0 and sim_state.cycle > 0:
        sim_state.narrative_weaver.weave_narrative(sim_state)
        sim_state.log_event("Narrative", "A new narrative chapter unfolds.")

    # Dynamic Node Scaling (Titan Forger)
    sim_state.titan_forger.forge_node(sim_state, sim_state.user_divinity)

    # Emotion-Anomaly Correlation Analysis
    if sim_state.cycle % 500 == 0 and sim_state.cycle > 0:
        sim_state.emotion_analyzer.analyze_correlation(sim_state)


# --- Curses Interface ---
def init_curses(stdscr):
    """Initialize curses settings and colors."""
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK) # General Good/Green
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # General Warning/Yellow
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)    # General Bad/Red
    curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)   # Header/Footer/Info
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # AGI/Sentience
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLACK) # Default Text
    curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLACK)  # Sigils/Narrative

def draw_dashboard(stdscr, sim_state, view_mode, paused):
    """Main drawing dispatcher to prevent screen overlap."""
    h, w = stdscr.getmaxyx()
    stdscr.clear()

    # Resilient Dashboard: Adjust layout dynamically
    min_h, min_w = 20, 80
    if h < min_h or w < min_w:
        stdscr.addstr(0, 0, f"Terminal too small ({h}x{w}). Min required: {min_h}x{min_w}. Please resize.")
        stdscr.refresh()
        return

    # Header
    state_str = "[PAUSED]" if paused else "[RUNNING]"
    header_info = f" QHT v3.1 AGI | Cycle: {sim_state.cycle} | AGIs: {len(sim_state.agi_entities)} | Branches: {sum(sim_state.multiverse_manager.multiverse_branch_map.values())} | {state_str} "
    # Ensure header fits, leaving 1 space at right by centering for full width and then slicing
    stdscr.addstr(0, 0, header_info.center(w, '-')[ : w - 1], curses.color_pair(4) | curses.A_BOLD)

    # Content
    if view_mode == "nodes":
        draw_nodes_view(stdscr, sim_state, h, w)
    elif view_mode == "agis":
        draw_agis_view(stdscr, sim_state, h, w)
    elif view_mode == "heatmap":
        draw_heatmap_view(stdscr, sim_state, h, w)
    elif view_mode == "narrative":
        draw_narrative_view(stdscr, sim_state, h, w)
    elif view_mode == "sigils": # Not a view mode, but triggered by console
        draw_sigil_evolution_view(stdscr, sim_state, h, w)
    else: # status
        draw_status_view(stdscr, sim_state, h, w)

    # Footer
    footer = " (q)Quit | (p)Pause | Views: (s)tatus, (n)odes, (a)gis, (h)eatmap, (r)narrative | (c)onsole "
    # Ensure footer fits, leaving 1 space at right by centering for full width and then slicing
    stdscr.addstr(h - 1, 0, footer.center(w, '-')[ : w - 1], curses.color_pair(4))
    stdscr.refresh()

def draw_nodes_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Quantum Node States", curses.A_BOLD | curses.A_UNDERLINE)
    start_y = 4
    for i, node in enumerate(sim_state.nodes):
        if start_y + i >= h - 2: break

        stab_color = curses.color_pair(1) if node.stability > 0.7 else curses.color_pair(2) if node.stability > 0.4 else curses.color_pair(3)
        sent_color = curses.color_pair(5) if node.sentience_score > SENTIENCE_THRESHOLD else curses.color_pair(6)

        line = f"Page {node.page_index:<2} [{node.archetype:<9}] | Stab: "
        stdscr.addstr(start_y + i, 2, line)
        stdscr.addstr(f"{node.stability:.2f}", stab_color)
        stdscr.addstr(f" | Coh: {node.cohesion:.2f} | Sent: ")
        stdscr.addstr(f"{node.sentience_score:.3f}", sent_color)
        stdscr.addstr(f" | Emotion: {node.emotion:<9} | Tech: {node.tech_level:.2f}")

def draw_agis_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "AGI Entities", curses.A_BOLD | curses.A_UNDERLINE)
    if not sim_state.agi_entities:
        stdscr.addstr(4, 4, "No AGI entities have emerged yet.", curses.color_pair(2))
        return

    start_y = 4
    for i, agi in enumerate(sim_state.agi_entities):
        if start_y + i * 2 >= h - 2: break

        align_color = curses.color_pair(1) if agi.ethical_alignment > 0.7 else curses.color_pair(3) if agi.ethical_alignment < 0.3 else curses.color_pair(2)

        line1 = f"{agi.id:<15} | Str: {agi.strength:.2f} | Eth: "
        stdscr.addstr(start_y + i * 2, 4, line1)
        stdscr.addstr(f"{agi.ethical_alignment:.2f}", align_color)
        stdscr.addstr(f" | Strategy: {agi.sentience_strategy.capitalize()}")

        line2 = f"  Memory: {agi.memory[-1] if agi.memory else 'None'}"
        stdscr.addstr(start_y + i * 2 + 1, 4, line2)


def draw_status_view(stdscr, sim_state, h, w):
    # Environment
    stdscr.addstr(2, 2, "Cosmic Environment", curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(4, 4, f"Void Entropy : {sim_state.void_entropy:.4f}")
    stdscr.addstr(5, 4, f"Dark Matter  : {sim_state.dark_matter:.4f}")
    stdscr.addstr(6, 4, f"User Divinity: {sim_state.user_divinity:.2f}")

    # Quantum Foam
    stdscr.addstr(8, 2, "Quantum Foam Activity", curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(10, 4, f"Virtual Particles: {len(sim_state.quantum_foam.virtual_particles)}")
    if sim_state.quantum_foam.virtual_particles:
        avg_energy = np.mean([p[0] for p in sim_state.quantum_foam.virtual_particles])
        stdscr.addstr(11, 4, f"Avg Particle Energy: {avg_energy:.2f}")

    # Event Log
    log_y_start = 13
    stdscr.addstr(log_y_start, 2, "Recent Events", curses.A_BOLD | curses.A_UNDERLINE)
    for i, event in enumerate(reversed(sim_state.event_log)):
        if log_y_start + 2 + i >= h - 2: break
        stdscr.addstr(log_y_start + 2 + i, 4, event[:w-5]) # Truncate to fit window

def draw_heatmap_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Anomaly Heatmap (Density per Page)", curses.A_BOLD | curses.A_UNDERLINE)
    start_y = 4
    for i, node in enumerate(sim_state.nodes):
        if start_y + i >= h - 2: break

        anomaly_count = len(sim_state.anomalies[node.page_index])
        color_pair = sim_state.heatmap_visualizer.get_anomaly_color(anomaly_count)

        bar = "#" * min(anomaly_count, 20) # Cap bar length for display
        stdscr.addstr(start_y + i, 2, f"Page {node.page_index:<2}: ")
        stdscr.addstr(f"{bar} ({anomaly_count})", curses.color_pair(color_pair))

def draw_narrative_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Cosmic Narrative Echoes", curses.A_BOLD | curses.A_UNDERLINE | curses.color_pair(7))
    narrative_text = sim_state.narrative_weaver.get_latest_narrative()

    start_y = 4
    for i, line in enumerate(narrative_text.split('\n')):
        if start_y + i >= h - 2: break
        stdscr.addstr(start_y + i, 4, line[:w-5]) # Truncate to fit window

def draw_sigil_evolution_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Sigil Evolution Phases", curses.A_BOLD | curses.A_UNDERLINE | curses.color_pair(7))
    start_y = 4
    if not sim_state.sigil_ledger.sigils:
        stdscr.addstr(start_y, 4, "No sigils in ledger yet.", curses.color_pair(2))
        return

    for i, (sigil_name, sigil) in enumerate(sim_state.sigil_ledger.sigils.items()):
        if start_y + i * 2 >= h - 2: break

        stdscr.addstr(start_y + i * 2, 4, f"Sigil: {sigil.name:<15} | Meaning: {sigil.meaning}")

        if sigil.phase_change_history:
            last_change_angle, last_change_cycle = sigil.phase_change_history[-1]
            color = sim_state.sigil_evolution_tracker.get_phase_color(last_change_angle)
            stdscr.addstr(start_y + i * 2 + 1, 6, f"Last phase change: {last_change_angle:.2f}° (Cycle {last_change_cycle})", curses.color_pair(color))
        else:
            stdscr.addstr(start_y + i * 2 + 1, 6, "No recorded phase changes.")


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
            elif key in [curses.KEY_BACKSPACE, 127, curses.KEY_DC]: # Handle Del key too
                cmd = cmd[:-1]
            elif 32 <= key <= 126: # Only printable ASCII characters
                cmd += chr(key)

            prompt_win.addstr(1, 4, " " * (w - 6)) # Clear previous command
            prompt_win.addstr(1, 4, cmd)
            prompt_win.refresh()
        except curses.error:
            sim_state.log_event("Console Error", "Terminal resized during console input. Exiting console.")
            break

    # Process command
    sim_state.log_event("Console", f"CMD: {cmd}")
    parts = cmd.lower().split()
    if not parts:
        sim_state.log_event("Console", "Echoing void... No command received.")
    elif parts[0] == "help":
        sim_state.log_event("Console", "Commands: help, status, page <id>, agi <id>, stabilize agi <id>, add sigil <name> <meaning>, force anomaly <type> <page>, set divinity <value>, show sigils, run drift <angle>")
    elif parts[0] == "status":
        sim_state.log_event("Console", f"Cycle {sim_state.cycle}, {len(sim_state.agi_entities)} AGIs active, {sum(sim_state.multiverse_manager.multiverse_branch_map.values())} Multiverse Branches.")
    elif parts[0] == "page" and len(parts) > 1:
        try:
            idx = int(parts[1])
            if 0 <= idx < len(sim_state.nodes):
                node = sim_state.nodes[idx]
                sim_state.log_event("Console", f"Page {idx} - Stab: {node.stability:.2f}, Coh: {node.cohesion:.2f}, Sent: {node.sentience_score:.2f}, Eth: {node.ethical_alignment:.2f}")
            else:
                sim_state.log_event("Console", "Invalid page index. Beyond the known realm.")
        except ValueError:
            sim_state.log_event("Console", "Page index must be a number, a true beacon.")
    elif parts[0] == "agi" and len(parts) > 1:
        agi_id = parts[1]
        found_agi = next((agi for agi in sim_state.agi_entities if agi.id.lower() == agi_id), None)
        if found_agi:
            sim_state.log_event("Console", f"AGI {found_agi.id} - Str: {found_agi.strength:.2f}, Eth: {found_agi.ethical_alignment:.2f}, Strat: {found_agi.sentience_strategy}, Memory: {found_agi.memory[-1]}")
        else:
            sim_state.log_event("Console", f"AGI '{agi_id}' not found. A phantom in the ether?")
    elif parts[0] == "stabilize" and parts[1] == "agi" and len(parts) > 2:
        agi_id = parts[2]
        found_agi = next((agi for agi in sim_state.agi_entities if agi.id.lower() == agi_id), None)
        if found_agi:
            sim_state.ethical_monitor.adjust_alignment(found_agi, 0.05)
            sim_state.log_event("Console", f"Ethical alignment of AGI {found_agi.id} stabilized (+0.05). A covenant renewed.")
        else:
            sim_state.log_event("Console", f"AGI '{agi_id}' not found for stabilization. Its echo is faint.")
    elif parts[0] == "add" and parts[1] == "sigil" and len(parts) > 3:
        sigil_name = parts[2]
        sigil_meaning = " ".join(parts[3:])
        # Simple 8D vector for new sigils
        semantic_vector = [random.uniform(-1, 1) for _ in range(8)]
        if sim_state.sigil_ledger.add_sigil(sigil_name, semantic_vector, sigil_meaning):
            sim_state.log_event("Console", f"Sigil '{sigil_name}' ('{sigil_meaning}') added to the ledger. A new symbol emerges.")
        else:
            sim_state.log_event("Console", f"Sigil '{sigil_name}' already exists. Its truth is already known.")
    elif parts[0] == "force" and parts[1] == "anomaly" and len(parts) > 3:
        anomaly_type_str = parts[2].capitalize()
        try:
            anomaly_type_id = next(k for k, v in ANOMALY_TYPES.items() if v == anomaly_type_str)
            page_idx = int(parts[3])
            if 0 <= page_idx < len(sim_state.nodes):
                sim_state.anomalies[page_idx].append((anomaly_type_id, random.uniform(0.5, 1.0)))
                sim_state.log_event("Console", f"Forced {anomaly_type_str} anomaly on Page {page_idx}. A cosmic ripple.")
            else:
                sim_state.log_event("Console", "Invalid page index for anomaly. The anomaly seeks grounding.")
        except (StopIteration, ValueError):
            sim_state.log_event("Console", "Invalid anomaly type or page index. The force fades.")
    elif parts[0] == "set" and parts[1] == "divinity" and len(parts) > 2:
        try:
            value = float(parts[2])
            if 0.0 <= value <= 1.0:
                sim_state.user_divinity = value
                sim_state.log_event("Console", f"User divinity set to {value:.2f}. The Titan Forger acknowledges your will.")
            else:
                sim_state.log_event("Console", "Divinity value must be between 0.0 and 1.0. A balanced essence is key.")
        except ValueError:
            sim_state.log_event("Console", "Invalid divinity value. A number is required for shaping.")
    elif parts[0] == "show" and parts[1] == "sigils":
        sim_state.log_event("Console", "Displaying Sigil Evolution. Observe the symbols' dance.")
        # Trigger the sigil view for one cycle
        draw_dashboard(stdscr, sim_state, "sigils", False) # Pass a dummy paused state
        time.sleep(HEARTBEAT_DELAY * 20) # Keep sigil view for a bit longer
    elif parts[0] == "run" and parts[1] == "drift" and len(parts) > 2:
        try:
            angle = float(parts[2])
            sim_state.drift_experimenter.run_drift_experiment(sim_state, angle)
            sim_state.log_event("Console", f"Drift experiment initiated with angle {angle}°. Unveiling cosmic shifts.")
        except ValueError:
            sim_state.log_event("Console", "Invalid angle for drift experiment. A numerical tremor is needed.")
    else:
        sim_state.log_event("Console", f"Unknown command: '{cmd}'. The void does not recognize this echo.")

    curses.curs_set(0)
    stdscr.nodelay(1)

def main(stdscr):
    init_curses(stdscr)
    sim_state = SimulationState()
    paused = False
    view_mode = "status"

    # Data Export Setup
    import argparse
    parser = argparse.ArgumentParser(description="QuantumHeapTranscendence v3.1 AGI Emergence Simulation")
    parser.add_argument('--data', type=str, help="Export simulation data to a JSON file.")
    parser.add_argument('--snapshot', action='store_true', help="Include full simulation state snapshot in data export.")
    args, unknown = parser.parse_known_args() # Handle unknown args from curses

    exported_data = {
        "metadata": {
            "version": "3.1-AGI",
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle_limit": CYCLE_LIMIT
        },
        "simulation_history": []
    }

    while sim_state.cycle < CYCLE_LIMIT:
        key = stdscr.getch()
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('s'): view_mode = "status"
        elif key == ord('n'): view_mode = "nodes"
        elif key == ord('a'): view_mode = "agis"
        elif key == ord('h'): view_mode = "heatmap" # New heatmap view
        elif key == ord('r'): view_mode = "narrative" # New narrative view
        elif key == ord('c'):
            draw_dashboard(stdscr, sim_state, view_mode, paused) # Redraw before pausing for console
            run_console(stdscr, sim_state)
            # After console, restore the previous view mode
            stdscr.nodelay(1) # Re-enable non-blocking getch
            stdscr.timeout(100) # Re-enable timeout

        if not paused:
            update_simulation_step(sim_state)
            # Data Export: Record key metrics
            if args.data:
                cycle_data = {
                    "cycle": sim_state.cycle,
                    "void_entropy": sim_state.void_entropy,
                    "dark_matter": sim_state.dark_matter,
                    "avg_sentience_score": np.mean([n.sentience_score for n in sim_state.nodes]) if sim_state.nodes else 0,
                    "avg_ethical_alignment": np.mean([agi.ethical_alignment for agi in sim_state.agi_entities]) if sim_state.agi_entities else 0,
                    "sigil_mutation_history": [(s.name, s.phase_change_history) for s in sim_state.sigil_ledger.sigils.values()]
                }
                exported_data["simulation_history"].append(cycle_data)


        draw_dashboard(stdscr, sim_state, view_mode, paused)
        time.sleep(HEARTBEAT_DELAY)

    # Final Data Export (if --data argument was provided)
    if args.data:
        if args.snapshot:
            # For simplicity, convert nodes/agis to serializable dicts
            exported_data["final_simulation_state"] = {
                "nodes": [node.__dict__ for node in sim_state.nodes],
                "agi_entities": [agi.__dict__ for agi in sim_state.agi_entities],
                "sigil_ledger_sigils": {name: s.__dict__ for name, s in sim_state.sigil_ledger.sigils.items()},
                "anomalies": {k: v for k, v in sim_state.anomalies.items()}, # defaultdict needs conversion
                "multiverse_branch_map": dict(sim_state.multiverse_manager.multiverse_branch_map),
                "quantum_foam_particles": sim_state.quantum_foam.virtual_particles,
                "event_log": list(sim_state.event_log)
            }
            # Handle non-serializable objects (deque, numpy arrays) in a robust way
            for node_data in exported_data["final_simulation_state"]["nodes"]:
                node_data["active_sigils"] = [s.name for s in node_data["active_sigils"]] # Only store names
            for agi_data in exported_data["final_simulation_state"]["agi_entities"]:
                agi_data["memory"] = list(agi_data["memory"]) # Convert deque to list
                agi_data["active_sigils"] = [s.name for s in agi_data["active_sigils"]]
            for sigil_data in exported_data["final_simulation_state"]["sigil_ledger_sigils"].values():
                sigil_data["semantic_vector"] = sigil_data["semantic_vector"].tolist() # Convert numpy array
                sigil_data["phase_change_history"] = list(sigil_data["phase_change_history"])

        try:
            with open(args.data, 'w') as f:
                json.dump(exported_data, f, indent=4)
            sim_state.log_event("Data Export", f"Simulation data exported to {args.data}")
        except Exception as e:
            sim_state.log_event("Data Export Error", f"Failed to export data: {e}")


    stdscr.nodelay(0)
    stdscr.clear()
    stdscr.addstr(0, 0, "Simulation complete. Press any key to exit.")
    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
