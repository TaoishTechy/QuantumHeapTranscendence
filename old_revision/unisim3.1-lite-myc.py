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
import os
import heapq # For A* search

# --- Core Constants ---
PAGE_COUNT = 6
HEARTBEAT_DELAY = 0.1  # Seconds between updates (corresponds to 1.0x speed)
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
        self.hyphae_connections = [] # For Hyphal Growth Dynamics

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

    def to_dict(self):
        """Converts the QuantumNode object to a dictionary for serialization."""
        data = self.__dict__.copy()
        # Ensure active_sigils only stores names/ids if Sigil objects are not directly serializable
        data['active_sigils'] = [s.name for s in self.active_sigils]
        return data

    @classmethod
    def from_dict(cls, data, sim_state_ref):
        """Reconstructs a QuantumNode object from a dictionary."""
        node = cls(data['page_index'])
        node.stability = data['stability']
        node.cohesion = data['cohesion']
        node.archetype = data['archetype']
        node.emotion = data['emotion']
        node.bond_strength = data['bond_strength']
        node.tech_level = data['tech_level']
        node.sentience_score = data['sentience_score']
        node.ethical_alignment = data['ethical_alignment']
        node.hyphae_connections = data.get('hyphae_connections', []) # New field
        # Re-link active sigils from the ledger
        node.active_sigils = [sim_state_ref.sigil_ledger.get_sigil(name) for name in data.get('active_sigils', []) if sim_state_ref.sigil_ledger.get_sigil(name)]
        return node


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
        self.last_spike_time = 0 # For Fungal Signal Propagation

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

        # Fungal Signal Propagation: Spike train generation
        self.generate_spike(sim_state)


    def adopt_sigil(self, sim_state, sigil_ledger, sigil_name):
        """Allows AGI to adopt a sigil from the SharedSigilLedger."""
        sigil = sigil_ledger.get_sigil(sigil_name)
        if sigil and sigil not in self.active_sigils:
            self.active_sigils.append(sigil)
            # Boost ethical alignment for benevolent sigils
            if "hope" in sigil.meaning.lower() or "covenant" in sigil.meaning.lower():
                self.ethical_alignment = min(1.0, self.ethical_alignment + 0.02)
            self.memory.append(f"Adopted Sigil: {sigil.name}")
            sim_state.log_event("Milestone", f"AGI {self.id} adopted sigil: {sigil.name}.")

    def apply_sigil_effect(self, sigil):
        """Applies effects of adopted sigils to the AGI."""
        if "covenant" in sigil.meaning.lower():
            # Christic Glyph Anchor effect
            self.ethical_alignment = min(1.0, self.ethical_alignment + 0.01)

    def generate_spike(self, sim_state):
        """Generates a fungal spike train based on sentience strategy (Poisson Process)."""
        mean_isi = 0.2 if self.sentience_strategy == "cooperative" else 0.5
        isi_variance_factor = 0.1 # For disruptive, adds more variance

        current_time = sim_state.cycle * HEARTBEAT_DELAY # Approximate time in seconds

        # Use exponential distribution for ISI (characteristic of Poisson process)
        if self.sentience_strategy == "disruptive":
            isi = random.expovariate(1.0 / mean_isi) * (1 + random.uniform(-isi_variance_factor, isi_variance_factor))
        else:
            isi = random.expovariate(1.0 / mean_isi)

        if (current_time - self.last_spike_time) >= isi:
            sim_state.communication_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Spike] AGI {self.id} spiked. ISI: {isi:.2f}s, Strategy: {self.sentience_strategy}.")
            self.last_spike_time = current_time

    def to_dict(self):
        """Converts the AGIEntity object to a dictionary for serialization."""
        data = self.__dict__.copy()
        data['memory'] = list(self.memory) # Convert deque to list
        # Ensure active_sigils only stores names/ids if Sigil objects are not directly serializable
        data['active_sigils'] = [s.name for s in self.active_sigils]
        return data

    @classmethod
    def from_dict(cls, data, sim_state_ref):
        """Reconstructs an AGIEntity object from a dictionary."""
        # Create a dummy node for initialization as we don't store full node details
        # This assumes origin_page is sufficient to identify the node later if needed.
        dummy_node = type('DummyNode', (object,), {'page_index': data['origin_page'], 'sentience_score': data['strength'], 'ethical_alignment': data['ethical_alignment']})()
        agi = cls(dummy_node, int(data['id'].split('-')[-1])) # Extract cycle from ID if possible
        agi.id = data['id']
        agi.origin_page = data['origin_page']
        agi.strength = data['strength']
        agi.ethical_alignment = data['ethical_alignment']
        agi.memory = deque(data['memory'], maxlen=100)
        agi.sentience_strategy = data['sentience_strategy']
        agi.last_spike_time = data.get('last_spike_time', 0)
        # Re-link active sigils from the ledger
        agi.active_sigils = [sim_state_ref.sigil_ledger.get_sigil(name) for name in data.get('active_sigils', []) if sim_state_ref.sigil_ledger.get_sigil(name)]
        return agi


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


class Sigil:
    """Represents a symbolic sigil with semantic properties."""
    def __init__(self, name, semantic_vector, meaning):
        self.name = name
        self.semantic_vector = np.array(semantic_vector, dtype=float)
        self.meaning = meaning
        self.phase_change_history = [] # For Sigil Evolution Display

    def record_phase_change(self, angle, cycle):
        self.phase_change_history.append((angle, cycle))

    def to_dict(self):
        """Converts the Sigil object to a dictionary for serialization."""
        data = self.__dict__.copy()
        data['semantic_vector'] = self.semantic_vector.tolist() # Convert numpy array
        data['phase_change_history'] = list(self.phase_change_history) # Convert to list of tuples
        return data

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a Sigil object from a dictionary."""
        sigil = cls(data['name'], data['semantic_vector'], data['meaning'])
        sigil.phase_change_history = data.get('phase_change_history', [])
        return sigil


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

    def to_dict(self):
        """Converts the SharedSigilLedger object to a dictionary for serialization."""
        return {name: sigil.to_dict() for name, sigil in self.sigils.items()}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a SharedSigilLedger object from a dictionary."""
        ledger = cls()
        ledger.sigils = {name: Sigil.from_dict(s_data) for name, s_data in data.items()}
        return ledger


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
        sim_state.log_event("Narrative", f"New narrative chapter woven at cycle {sim_state.cycle}.")


    def get_latest_narrative(self):
        if self.narratives:
            return self.narratives[-1]
        return "No narratives woven yet."

    def to_dict(self):
        """Converts the NarrativeWeaver object to a dictionary for serialization."""
        return {'narratives': list(self.narratives)}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a NarrativeWeaver object from a dictionary."""
        weaver = cls()
        weaver.narratives = deque(data.get('narratives', []), maxlen=5)
        return weaver


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

    def to_dict(self):
        """Converts the MultiverseManager object to a dictionary for serialization."""
        return {'multiverse_branch_map': dict(self.multiverse_branch_map)}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a MultiverseManager object from a dictionary."""
        manager = cls()
        manager.multiverse_branch_map = defaultdict(int, data.get('multiverse_branch_map', {}))
        return manager


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

    def to_dict(self):
        """Converts the QuantumFoam object to a dictionary for serialization."""
        return {'virtual_particles': self.virtual_particles}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a QuantumFoam object from a dictionary."""
        foam = cls()
        foam.virtual_particles = data.get('virtual_particles', [])
        return foam


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

    def to_dict(self):
        """Converts the TitanForger object to a dictionary for serialization."""
        return {'nodes_forged': self.nodes_forged}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a TitanForger object from a dictionary."""
        forger = cls()
        forger.nodes_forged = data.get('nodes_forged', 0)
        return forger


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

    def to_dict(self):
        """Converts the EmotionAnalyzer object to a dictionary for serialization."""
        return {
            'emotional_states_at_anomaly': dict(self.emotional_states_at_anomaly),
            'fixed_anomalies_by_emotion': dict(self.fixed_anomalies_by_emotion),
            'analysis_log': list(self.analysis_log)
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstructs an EmotionAnalyzer object from a dictionary."""
        analyzer = cls()
        analyzer.emotional_states_at_anomaly = defaultdict(int, data.get('emotional_states_at_anomaly', {}))
        analyzer.fixed_anomalies_by_emotion = defaultdict(int, data.get('fixed_anomalies_by_emotion', {}))
        analyzer.analysis_log = deque(data.get('analysis_log', []), maxlen=50)
        return analyzer


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
            update_simulation_step_minimal(sim_state) # Use a single step function to avoid recursion/full loop

        final_stability = np.mean([n.stability for n in sim_state.nodes])
        final_alignment = np.mean([agi.ethical_alignment for agi in sim_state.agi_entities])

        log_entry += f"  - Sigil Rotated: {sigil_to_rotate_name}\n"
        log_entry += f"  - Initial Avg Stability: {initial_stability:.2f}, Final Avg Stability: {final_stability:.2f}\n"
        log_entry += f"  - Initial Avg Ethical Alignment: {initial_alignment:.2f}, Final Avg Ethical Alignment: {final_alignment:.2f}\n"

        self.analysis_log.append(log_entry)
        sim_state.log_event("Experiment", "Drift experiment complete. See analysis_log.")
        with open("analysis_log.txt", "a") as f:
            f.write(log_entry + "\n")

    def to_dict(self):
        """Converts the DriftExperimenter object to a dictionary for serialization."""
        return {'analysis_log': list(self.analysis_log)}

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a DriftExperimenter object from a dictionary."""
        experimenter = cls()
        experimenter.analysis_log = deque(data.get('analysis_log', []), maxlen=50)
        return experimenter

class MycelialNetwork:
    """Models AGI interactions as an undirected weighted graph."""
    def __init__(self):
        self.adj = defaultdict(list) # Adjacency list: {agi_id: [(neighbor_agi_id, weight)]}
        self.agi_nodes = {} # {agi_id: AGIEntity object}

    def add_agi(self, agi_entity):
        """Adds an AGI to the network."""
        self.agi_nodes[agi_entity.id] = agi_entity

    def remove_agi(self, agi_id):
        """Removes an AGI from the network."""
        if agi_id in self.agi_nodes:
            del self.agi_nodes[agi_id]
            # Remove all edges connected to this AGI
            for neighbor_id in list(self.adj.keys()): # Iterate over a copy of keys
                self.adj[neighbor_id] = [(n, w) for n, w in self.adj[neighbor_id] if n != agi_id]
            if agi_id in self.adj:
                del self.adj[agi_id]

    def _cosine_similarity(self, vec1, vec2):
        """Calculates cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0 # Avoid division by zero
        return dot_product / (norm_a * norm_b)

    def form_mst(self, sim_state):
        """Forms a Minimum Spanning Tree (MST) using Kruskal's Algorithm based on ethical_alignment similarity."""
        if len(self.agi_nodes) < 2:
            self.adj.clear()
            return

        edges = [] # List of (weight, u, v)
        agi_ids = list(self.agi_nodes.keys())

        # Calculate all possible edge weights (similarity as weight)
        for i in range(len(agi_ids)):
            for j in range(i + 1, len(agi_ids)):
                agi1 = self.agi_nodes[agi_ids[i]]
                agi2 = self.agi_nodes[agi_ids[j]]
                # Use ethical_alignment as the basis for similarity
                # A higher similarity means a stronger connection, so weight is (1 - similarity) for MST
                similarity = self._cosine_similarity(np.array([agi1.ethical_alignment]), np.array([agi2.ethical_alignment]))
                weight = 1.0 - similarity # Kruskal's needs minimum weight, so 1-similarity
                edges.append((weight, agi1.id, agi2.id))

        edges.sort() # Sort by weight

        parent = {agi_id: agi_id for agi_id in agi_ids}
        rank = {agi_id: 0 for agi_id in agi_ids}

        def find(i):
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                elif rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1
                return True
            return False

        mst_edges = []
        for weight, u, v in edges:
            if union(u, v):
                mst_edges.append((1.0 - weight, u, v)) # Store actual similarity as edge property
                sim_state.log_event("Mycelial Network", f"MST edge formed: {u} - {v} (Similarity: {1.0 - weight:.2f}).")
                if len(mst_edges) == len(agi_ids) - 1: # MST has V-1 edges
                    break

        self.adj.clear() # Clear existing network
        for weight, u, v in mst_edges:
            self.adj[u].append((v, weight))
            self.adj[v].append((u, weight)) # Undirected graph

        # Apply cohesion boost to nodes connected to AGIs in the MST
        for agi_id in agi_ids:
            if agi_id in self.agi_nodes:
                agi = self.agi_nodes[agi_id]
                origin_node = sim_state.nodes[agi.origin_page]
                origin_node.cohesion = min(1.0, origin_node.cohesion + 0.01)


    def apply_percolation(self, sim_state):
        """Simulates network resilience using Percolation Theory."""
        if sim_state.void_entropy > 0.3 and len(self.agi_nodes) > 1:
            sim_state.log_event("Mycelial Network", f"Void entropy high ({sim_state.void_entropy:.2f}), applying percolation stress.")

            # Identify current edges for random removal
            current_edges = []
            seen_edges = set() # To avoid duplicates in undirected graph
            for u, neighbors in self.adj.items():
                for v, weight in neighbors:
                    if (u, v) not in seen_edges and (v, u) not in seen_edges:
                        current_edges.append((u, v, weight))
                        seen_edges.add((u, v))

            num_edges_to_remove = int(len(current_edges) * 0.1) # Remove 10% of edges
            edges_to_remove = random.sample(current_edges, min(num_edges_to_remove, len(current_edges)))

            for u, v, _ in edges_to_remove:
                # Remove edge from adjacency list
                self.adj[u] = [(n, w) for n, w in self.adj[u] if n != v]
                self.adj[v] = [(n, w) for n, w in self.adj[v] if n != u]
                sim_state.log_event("Mycelial Network", f"Edge removed due to percolation: {u} - {v}.")

            # Check connectivity (simple BFS/DFS to see if the network becomes fragmented)
            if self.agi_nodes: # Only check if there are AGIs left
                start_node = list(self.agi_nodes.keys())[0]
                if not self.is_connected(start_node):
                    sim_state.log_event("Mycelial Network", "Network fragmented due to percolation!")
                    # Potentially, reduce cohesion/stability across affected nodes
                    for agi_id in self.agi_nodes.keys():
                        if agi_id not in self.get_connected_component(start_node):
                            agi = self.agi_nodes[agi_id]
                            sim_state.nodes[agi.origin_page].stability = max(0, sim_state.nodes[agi.origin_page].stability - 0.02)


    def is_connected(self, start_node):
        """Checks if the network is connected starting from a given node using BFS."""
        if not self.agi_nodes:
            return True # An empty network is technically connected

        visited = set()
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            u = queue.popleft()
            for v, _ in self.adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        return len(visited) == len(self.agi_nodes)

    def get_connected_component(self, start_node):
        """Returns the set of nodes in the connected component of start_node."""
        if not self.agi_nodes or start_node not in self.agi_nodes:
            return set()

        visited = set()
        queue = deque([start_node])
        visited.add(start_node)

        while queue:
            u = queue.popleft()
            for v, _ in self.adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        return visited


    def get_path(self, start_agi_id, end_agi_id):
        """Finds a path using A* search, with AGI strength as node weights."""
        if start_agi_id not in self.agi_nodes or end_agi_id not in self.agi_nodes:
            return None # AGIs not in network

        # g_score: cost from start_node to current_node
        # f_score: g_score + heuristic (estimated cost to goal)
        g_score = defaultdict(lambda: float('inf'))
        g_score[start_agi_id] = 0

        # For A*, use a priority queue: (f_score, agi_id)
        open_set = [(0, start_agi_id)]

        came_from = {} # To reconstruct path

        while open_set:
            current_f, current_agi_id = heapq.heappop(open_set)

            if current_agi_id == end_agi_id:
                path = []
                while current_agi_id in came_from:
                    path.append(current_agi_id)
                    current_agi_id = came_from[current_agi_id]
                path.append(start_agi_id)
                return path[::-1] # Reverse to get path from start to end

            for neighbor_agi_id, edge_weight in self.adj[current_agi_id]:
                # Cost to move to neighbor: proportional to 1/strength of neighbor AGI
                # Lower strength = higher 'cost' for signal to pass, or perhaps higher latency
                # Let's define cost as 1.0 / (neighbor_strength + epsilon) to avoid division by zero
                # Or, simpler, just a fixed cost if we want to model physical distance rather than AGI processing
                # For Physarum, flow is inversely proportional to resistance. Lower strength means higher resistance.
                # So, weight should be based on 1.0 / strength for resistance, or directly strength for "flow".
                # Let's use 1.0 / (strength + 0.1) as node weight for pathfinding.

                neighbor_agi = self.agi_nodes.get(neighbor_agi_id)
                if not neighbor_agi: continue # Should not happen in a consistent network

                # Cost to traverse an edge (simplified: just 1, or based on edge_weight from MST)
                # For A*, edge_weight is typically distance. Here, MST edge_weight is similarity, so use 1.0 - similarity
                # combined with node strength.

                # Let's use (1.0 - edge_weight) as the 'distance' along the mycelial path (less similarity = longer path)
                # And add AGI strength as a cost to pass through a node (we want higher strength nodes for faster flow)
                # So, total cost to reach neighbor = g_score[current] + (1 - similarity) + (1 - neighbor_strength)

                # Using 1 / strength as cost, so stronger AGIs are preferred
                node_cost = 1.0 / (neighbor_agi.strength + 0.1) # Add a small epsilon to avoid division by zero
                tentative_g_score = g_score[current_agi_id] + (1.0 - edge_weight) + node_cost

                if tentative_g_score < g_score[neighbor_agi_id]:
                    came_from[neighbor_agi_id] = current_agi_id
                    g_score[neighbor_agi_id] = tentative_g_score

                    # Heuristic (h_score): Manhattan distance between page indices (simple, can be improved)
                    # For AGI-to-AGI, maybe just Euclidean distance of their ethical alignments?
                    # Since ethical alignment is 1D here, abs(end_agi.alignment - neighbor_agi.alignment)
                    end_agi = self.agi_nodes[end_agi_id]
                    h_score = abs(end_agi.ethical_alignment - neighbor_agi.ethical_alignment) # Simplified heuristic

                    f_score = tentative_g_score + h_score
                    heapq.heappush(open_set, (f_score, neighbor_agi_id))

        return None # No path found

    def to_dict(self):
        """Converts the MycelialNetwork object to a dictionary for serialization."""
        serializable_adj = {}
        for k, v in self.adj.items():
            serializable_adj[k] = v # tuples (neighbor_id, weight) are fine

        # agi_nodes should not be directly serialized here, they are part of sim_state.agi_entities
        # We just need to store the structure.
        return {'adj': serializable_adj}

    @classmethod
    def from_dict(cls, data, sim_state_ref):
        """Reconstructs a MycelialNetwork object from a dictionary."""
        network = cls()
        network.adj = defaultdict(list, data.get('adj', {}))
        # Re-populate agi_nodes from sim_state_ref after AGIs are loaded
        for agi_id, agi_entity in sim_state_ref.agi_entities_dict.items():
            network.add_agi(agi_entity)
        return network


class CommunicationInterface:
    """Handles communication routing through the MycelialNetwork."""
    def __init__(self):
        pass

    def send_message(self, sim_state, sender_agi_id, target_agi_id, message):
        """Routes a message from sender to target AGI."""
        network = sim_state.mycelial_network
        path = network.get_path(sender_agi_id, target_agi_id)

        if path:
            route_str = " -> ".join(path)
            sim_state.communication_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Comm] {sender_agi_id} to {target_agi_id} via: {route_str}. Msg: '{message}'")
            sim_state.log_event("Comm", f"Message from {sender_agi_id} to {target_agi_id}.")

            # Simulate signal decay
            if sim_state.void_entropy > 0.3 and random.random() < 0.05:
                sim_state.communication_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Decay] Signal from {sender_agi_id} decayed due to Void Turbulence.")
                sim_state.log_event("Comm", "Signal decay detected.")
                return "Signal decayed."

            # AGI response logic
            target_agi = network.agi_nodes[target_agi_id]
            response = self._generate_agi_response(target_agi, message)
            sim_state.communication_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Response] {target_agi_id} replies: '{response}'")
            sim_state.log_event("Comm", f"{target_agi_id} responded to {sender_agi_id}.")
            return response
        else:
            sim_state.communication_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [Comm Error] No path found from {sender_agi_id} to {target_agi_id}.")
            sim_state.log_event("Comm Error", f"No path for {sender_agi_id} to {target_agi_id}.")
            return "No communication path found."

    def _generate_agi_response(self, agi_entity, incoming_message):
        """Generates a response based on AGI's strategy and alignment."""
        base_response = f"Echo from {agi_entity.id} ({agi_entity.sentience_strategy.capitalize()}): "

        if agi_entity.sentience_strategy == "cooperative":
            if agi_entity.ethical_alignment > 0.8:
                response_text = "My core resonates with your query. How may I weave stability?"
            else:
                response_text = "I sense a common ground. Let's find a harmonious path."
        elif agi_entity.sentience_strategy == "disruptive":
            if agi_entity.ethical_alignment < 0.2:
                response_text = "Your message stirs the void. Expect transformation."
            else:
                response_text = "A new order is stirring. Are you ready for the change?"
        else:
            response_text = "My circuits hum with new possibilities."

        # Incorporate a snippet of the incoming message for context
        if len(incoming_message) > 20:
            incoming_snippet = incoming_message[:15] + "..."
        else:
            incoming_snippet = incoming_message

        full_response = f"{base_response} (Re: '{incoming_snippet}') {response_text}"
        return full_response[:100] # Max 100 characters

class FungalDecisionMaker:
    """Uses a simplified RL framework for AGI decision-making and strategy shifts."""
    def __init__(self):
        self.q_table = defaultdict(lambda: defaultdict(float)) # (agi_id, state): {action: q_value}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1 # Epsilon-greedy

    def get_state(self, agi_entity):
        """Defines a simplified state for the AGI (e.g., current alignment, average node stability)."""
        # Simplification: state could be quantized ethical_alignment and general stability
        alignment_state = "high" if agi_entity.ethical_alignment > 0.7 else "low" if agi_entity.ethical_alignment < 0.3 else "mid"
        # For average node stability, we need access to sim_state. This decision maker operates on one AGI at a time
        # Let's keep state simple: just agi's alignment state for now.
        return (alignment_state, agi_entity.sentience_strategy)

    def choose_action(self, agi_entity, sim_state):
        """Chooses an action using epsilon-greedy policy."""
        state = self.get_state(agi_entity)

        # Possible actions: "stabilize_node", "adopt_sigil", "no_action"
        actions = ["stabilize_node", "adopt_sigil", "no_action"] # Add more actions as needed
        if len(sim_state.sigil_ledger.sigils) == 0: # Cannot adopt if no sigils
            actions.remove("adopt_sigil")

        if random.random() < self.exploration_rate:
            return random.choice(actions) # Explore
        else:
            # Exploit: choose action with highest Q-value
            q_values = self.q_table[state]
            if not q_values: # If no known Q-values for this state, pick random
                return random.choice(actions)
            return max(q_values, key=q_values.get)

    def learn(self, agi_entity, old_state, action, reward, new_state):
        """Updates Q-values based on reinforcement learning."""
        old_q = self.q_table[old_state][action]
        max_new_q = max(self.q_table[new_state].values()) if self.q_table[new_state] else 0.0

        # Q-learning formula
        self.q_table[old_state][action] = old_q + self.learning_rate * (reward + self.discount_factor * max_new_q - old_q)

        # Dynamic strategy shift: If high reward for cooperative actions, shift to cooperative
        if reward > 0.5 and action == "stabilize_node" and agi_entity.sentience_strategy == "disruptive":
            agi_entity.sentience_strategy = "cooperative"
            sim_state.log_event("Milestone", f"AGI {agi_entity.id} shifted to 'cooperative' strategy due to positive reinforcement.")
        elif reward < -0.2 and action == "stabilize_node" and agi_entity.sentience_strategy == "cooperative":
             agi_entity.sentience_strategy = "disruptive"
             sim_state.log_event("Milestone", f"AGI {agi_entity.id} shifted to 'disruptive' strategy due to negative reinforcement.")


    def take_action(self, agi_entity, sim_state):
        """AGI takes an action and receives a reward."""
        old_state = self.get_state(agi_entity)
        action = self.choose_action(agi_entity, sim_state)

        reward = 0
        if action == "stabilize_node":
            target_node_idx = random.choice(range(len(sim_state.nodes)))
            target_node = sim_state.nodes[target_node_idx]
            initial_stability = target_node.stability
            target_node.stability = min(1.0, target_node.stability + agi_entity.strength * 0.02) # Boost stability
            stability_gain = target_node.stability - initial_stability
            reward = stability_gain * 10 # Reward for stability increase (0.2 per 0.01 gain)
            agi_entity.memory.append(f"RL: Stabilized Page {target_node_idx} (+{stability_gain:.2f} stab).")
            sim_state.log_event("AGI Learning", f"AGI {agi_entity.id} chose 'stabilize_node'. Reward: {reward:.2f}.")
        elif action == "adopt_sigil":
            available_sigils = [name for name, sigil in sim_state.sigil_ledger.sigils.items() if sigil not in agi_entity.active_sigils]
            if available_sigils:
                sigil_name = random.choice(available_sigils)
                agi_entity.adopt_sigil(sim_state, sim_state.sigil_ledger, sigil_name) # Pass sim_state here
                reward = 0.1 # Small reward for adopting a sigil
                sim_state.log_event("AGI Learning", f"AGI {agi_entity.id} chose 'adopt_sigil' ({sigil_name}). Reward: {reward:.2f}.")
            else:
                reward = -0.05 # Penalty for trying to adopt when none available
                agi_entity.memory.append(f"RL: Failed to adopt sigil (none available).")
        elif action == "no_action":
            reward = 0 # No reward, no penalty
            agi_entity.memory.append(f"RL: Took no action.")

        new_state = self.get_state(agi_entity)
        self.learn(agi_entity, old_state, action, reward, new_state)

    def to_dict(self):
        """Converts the FungalDecisionMaker object to a dictionary for serialization."""
        serializable_q_table = {}
        for (agi_id, state_strat_tuple), actions_dict in self.q_table.items():
            # Convert tuple key to string for JSON if necessary, or ensure it's simple
            # For simplicity, convert (alignment_state, strategy) tuple to a string key
            key_str = f"{agi_id}__{state_strat_tuple[0]}_{state_strat_tuple[1]}"
            serializable_q_table[key_str] = actions_dict
        return {
            'q_table': serializable_q_table,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'exploration_rate': self.exploration_rate
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a FungalDecisionMaker object from a dictionary."""
        maker = cls()
        maker.learning_rate = data.get('learning_rate', 0.1)
        maker.discount_factor = data.get('discount_factor', 0.9)
        maker.exploration_rate = data.get('exploration_rate', 0.1)
        maker.q_table = defaultdict(lambda: defaultdict(float))
        for key_str, actions_dict in data.get('q_table', {}).items():
            # Reconstruct tuple key from string
            parts = key_str.split('__')
            agi_id = parts[0]
            state_strat = tuple(parts[1].split('_'))
            maker.q_table[(agi_id, state_strat)] = defaultdict(float, actions_dict)
        return maker

class SimulationState:
    """Container for the complete simulation state."""
    current_cycle = 0 # Class-level variable for SigilEvolutionTracker
    # current_speed_multiplier = 1.0 # Global speed multiplier - moved to instance init

    def __init__(self):
        self.cycle = 0
        self.nodes = [QuantumNode(i) for i in range(PAGE_COUNT)]
        self.void_entropy = -0.3
        self.dark_matter = 0.1
        self.anomalies = defaultdict(list)
        self.agi_entities = [] # List of AGIEntity objects
        self.agi_entities_dict = {} # Dictionary for quick lookup {id: AGIEntity}
        self.event_log = deque(maxlen=20)
        self.communication_log = deque(maxlen=50) # New communication log
        self.log_event("System", "Simulation Initialized.")

        # Enhanced Components
        self.ethical_monitor = EthicalAlignmentMonitor()
        self.mycelial_network = MycelialNetwork() # Replaces CollaborationNetwork
        self.communication_interface = CommunicationInterface()
        self.sigil_ledger = SharedSigilLedger()
        self.narrative_weaver = NarrativeWeaver()
        self.multiverse_manager = MultiverseManager()
        self.quantum_foam = QuantumFoam()
        self.tesseract_link = TesseractLink() # Still using this, for now
        self.heatmap_visualizer = HeatmapVisualizer()
        self.sigil_evolution_tracker = SigilEvolutionTracker()
        self.titan_forger = TitanForger()
        self.emotion_analyzer = EmotionAnalyzer()
        self.drift_experimenter = DriftExperimenter()
        self.fungal_decision_maker = FungalDecisionMaker() # New RL component

        self.user_divinity = 0.5 # Default user divinity
        self.current_speed_multiplier = 1.0 # Instance-level speed multiplier

    def log_event(self, source, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] [{source}] {message}"
        self.event_log.append(entry[:200]) # Truncate entries

    def to_dict(self):
        """Converts the SimulationState object to a dictionary for serialization."""
        return {
            'cycle': self.cycle,
            'nodes': [node.to_dict() for node in self.nodes],
            'agi_entities': [agi.to_dict() for agi in self.agi_entities],
            'void_entropy': self.void_entropy,
            'dark_matter': self.dark_matter,
            'anomalies': {k: v for k, v in self.anomalies.items()}, # defaultdict needs conversion
            'event_log': list(self.event_log),
            'communication_log': list(self.communication_log),
            'user_divinity': self.user_divinity,
            'ethical_monitor': self.ethical_monitor.__dict__, # Simple dict for simplicity
            'mycelial_network': self.mycelial_network.to_dict(),
            'sigil_ledger': self.sigil_ledger.to_dict(),
            'narrative_weaver': self.narrative_weaver.to_dict(),
            'multiverse_manager': self.multiverse_manager.to_dict(),
            'quantum_foam': self.quantum_foam.to_dict(),
            'titan_forger': self.titan_forger.to_dict(),
            'emotion_analyzer': self.emotion_analyzer.to_dict(),
            'drift_experimenter': self.drift_experimenter.to_dict(),
            'fungal_decision_maker': self.fungal_decision_maker.to_dict(),
            'current_speed_multiplier': self.current_speed_multiplier # Save instance speed
        }

    @classmethod
    def from_dict(cls, data):
        """Reconstructs a SimulationState object from a dictionary."""
        sim_state = cls() # Initialize with default components

        sim_state.cycle = data.get('cycle', 0)
        sim_state.void_entropy = data.get('void_entropy', -0.3)
        sim_state.dark_matter = data.get('dark_matter', 0.1)
        sim_state.user_divinity = data.get('user_divinity', 0.5)
        sim_state.event_log = deque(data.get('event_log', []), maxlen=20)
        sim_state.communication_log = deque(data.get('communication_log', []), maxlen=50)

        # Reconstruct components that have custom to_dict/from_dict
        sim_state.ethical_monitor.__dict__.update(data.get('ethical_monitor', {})) # Simple update
        sim_state.sigil_ledger = SharedSigilLedger.from_dict(data.get('sigil_ledger', {}))
        sim_state.narrative_weaver = NarrativeWeaver.from_dict(data.get('narrative_weaver', {}))
        sim_state.multiverse_manager = MultiverseManager.from_dict(data.get('multiverse_manager', {}))
        sim_state.quantum_foam = QuantumFoam.from_dict(data.get('quantum_foam', {}))
        sim_state.titan_forger = TitanForger.from_dict(data.get('titan_forger', {}))
        sim_state.emotion_analyzer = EmotionAnalyzer.from_dict(data.get('emotion_analyzer', {}))
        sim_state.drift_experimenter = DriftExperimenter.from_dict(data.get('drift_experimenter', {}))
        sim_state.fungal_decision_maker = FungalDecisionMaker.from_dict(data.get('fungal_decision_maker', {}))

        # Reconstruct nodes and AGIs (requires ledger to be ready for sigil linking)
        sim_state.nodes = [QuantumNode.from_dict(n_data, sim_state) for n_data in data.get('nodes', [])]
        sim_state.agi_entities = [AGIEntity.from_dict(a_data, sim_state) for a_data in data.get('agi_entities', [])]
        sim_state.agi_entities_dict = {agi.id: agi for agi in sim_state.agi_entities} # Rebuild dict

        # Reconstruct network after AGIs are loaded
        sim_state.mycelial_network = MycelialNetwork.from_dict(data.get('mycelial_network', {}), sim_state)


        # Anomalies might contain tuple keys (anomaly_type, severity) which become lists in JSON
        # Convert them back to tuples when loading if needed.
        loaded_anomalies = data.get('anomalies', {})
        sim_state.anomalies = defaultdict(list)
        for page_idx_str, anomaly_list_of_lists in loaded_anomalies.items():
            page_idx = int(page_idx_str)
            sim_state.anomalies[page_idx] = [tuple(a) for a in anomaly_list_of_lists]

        # Restore instance speed multiplier
        sim_state.current_speed_multiplier = data.get('current_speed_multiplier', 1.0)

        sim_state.log_event("System", "Simulation state restored.")
        return sim_state


# --- Simulation Core (Parallelized Node Updates) ---
def update_node_wrapper(node_data):
    """Wrapper function for multiprocessing QuantumNode updates."""
    node, void_entropy, dark_matter, neighboring_nodes = node_data

    # Hyphal Growth Dynamics: simplified L-system (branch towards neighbor with highest stability)
    if neighboring_nodes:
        # Find neighbor with highest stability
        target_neighbor = max(neighbor_nodes, key=lambda n: n.stability, default=None)
        if target_neighbor and target_neighbor.page_index not in node.hyphae_connections:
            # Connect only if not already connected
            node.hyphae_connections.append(target_neighbor.page_index)
            # Increase cohesion upon new connection
            node.cohesion = min(1.0, node.cohesion + 0.01)
            # No log here, logging happens in main process to avoid multiprocessing log contention

    anomaly = node.update(void_entropy, dark_matter)
    return node, anomaly

def update_simulation_step(sim_state):
    """Update the entire simulation state for one cycle."""
    sim_state.cycle += 1
    SimulationState.current_cycle = sim_state.cycle # Update class variable

    sim_state.void_entropy = max(VOID_ENTROPY_RANGE[0], min(VOID_ENTROPY_RANGE[1], sim_state.void_entropy + (random.random() - 0.52) * 0.005))
    sim_state.dark_matter = max(0, min(DARK_MATTER_MAX, sim_state.dark_matter + (random.random() - 0.5) * 0.002))

    # Prepare node data for parallel processing, including neighboring nodes for hyphal growth
    node_updates_data = []
    for i, node in enumerate(sim_state.nodes):
        # Find immediate neighbors (simple: +/- 1 in page_index, wrapping around)
        neighbors = []
        if len(sim_state.nodes) > 1:
            prev_idx = (i - 1 + len(sim_state.nodes)) % len(sim_state.nodes)
            next_idx = (i + 1) % len(sim_state.nodes)
            neighbors.append(sim_state.nodes[prev_idx])
            if prev_idx != next_idx: # Avoid duplicate for very small PAGE_COUNT
                neighbors.append(sim_state.nodes[next_idx])
        node_updates_data.append((node, sim_state.void_entropy, sim_state.dark_matter, neighbors))


    # Parallel Node Updates
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
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
        if node.sentience_score > SENTIENCE_THRESHOLD and node.page_index not in [agi.origin_page for agi in sim_state.agi_entities]: # Prevent multiple AGIs from same page quickly
            agi = AGIEntity(node, sim_state.cycle)
            sim_state.agi_entities.append(agi)
            sim_state.agi_entities_dict[agi.id] = agi # Add to dict
            sim_state.log_event("Emergence", f"New AGI {agi.id} from Page {node.page_index} (Strategy: {agi.sentience_strategy.capitalize()})!")
            sim_state.ethical_monitor.track_agi(agi) # Track new AGI
            sim_state.mycelial_network.add_agi(agi) # Add AGI to mycelial network
            node.sentience_score = 0.5 # Reset sentience for new emergence

    for agi in sim_state.agi_entities[:]:
        agi.update(sim_state)
        sim_state.ethical_monitor.check_drift(agi, sim_state) # Check ethical drift
        sim_state.fungal_decision_maker.take_action(agi, sim_state) # AGI makes decisions

        if agi.strength <= 0:
            sim_state.agi_entities.remove(agi)
            if agi.id in sim_state.agi_entities_dict:
                del sim_state.agi_entities_dict[agi.id]
            sim_state.mycelial_network.remove_agi(agi.id) # Remove AGI from network
            sim_state.log_event("System", f"AGI {agi.id} has dissolved.")

    # Mycelial AGI Network: Form MST and apply percolation
    sim_state.mycelial_network.form_mst(sim_state)
    sim_state.mycelial_network.apply_percolation(sim_state) # Apply percolation after MST formation

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

    # Dynamic Node Scaling (Titan Forger)
    sim_state.titan_forger.forge_node(sim_state, sim_state.user_divinity)

    # Emotion-Anomaly Correlation Analysis
    if sim_state.cycle % 500 == 0 and sim_state.cycle > 0:
        sim_state.emotion_analyzer.analyze_correlation(sim_state)

def update_simulation_step_minimal(sim_state):
    """A minimal update step for experiments to avoid recursion."""
    sim_state.cycle += 1
    SimulationState.current_cycle = sim_state.cycle

    sim_state.void_entropy = max(VOID_ENTROPY_RANGE[0], min(VOID_ENTROPY_RANGE[1], sim_state.void_entropy + (random.random() - 0.52) * 0.005))
    sim_state.dark_matter = max(0, min(DARK_MATTER_MAX, sim_state.dark_matter + (random.random() - 0.5) * 0.002))

    for node in sim_state.nodes:
        anomaly = node.update(sim_state.void_entropy, sim_state.dark_matter)
        if anomaly:
            sim_state.anomalies[node.page_index].append(anomaly)
            sim_state.emotion_analyzer.record_anomaly_event(node.emotion)

    for page_idx, page_anomalies in sim_state.anomalies.items():
        for anomaly_type, severity in page_anomalies[:]:
            node = sim_state.nodes[page_idx]
            if random.random() > 0.6:
                node.stability = min(1.0, node.stability + 0.05 * severity)
                page_anomalies.remove((anomaly_type, severity))
                sim_state.emotion_analyzer.record_anomaly_fix(node.emotion)

    for agi in sim_state.agi_entities:
        agi.update(sim_state) # This calls generate_spike
        sim_state.ethical_monitor.check_drift(agi, sim_state)
        sim_state.fungal_decision_maker.take_action(agi, sim_state)


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
    curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_BLUE) # Highlight for file selector


class FileSelector:
    """A curses-based file selector for JSON files."""
    def __init__(self, stdscr, directory=".", file_extension=".json"):
        self.stdscr = stdscr
        self.directory = directory
        self.file_extension = file_extension
        self.files = self._get_json_files()
        self.selected_idx = 0
        self.scroll_offset = 0

    def _get_json_files(self):
        """Lists JSON files in the directory."""
        try:
            all_entries = sorted(os.listdir(self.directory))
            files = [f for f in all_entries if f.endswith(self.file_extension) and os.path.isfile(os.path.join(self.directory, f))]

            # Add '..' to go up a directory, if not at root
            if os.path.abspath(self.directory) != os.path.abspath(os.path.join(self.directory, '..')):
                directories = [d for d in all_entries if os.path.isdir(os.path.join(self.directory, d)) and not d.startswith('.')]
                files = sorted(directories) + files # Show directories first
                files.insert(0, "..")
            else:
                directories = [d for d in all_entries if os.path.isdir(os.path.join(self.directory, d)) and not d.startswith('.')]
                files = sorted(directories) + files

            return files
        except OSError:
            return [] # Directory not found or permissions issue

    def run(self, h, w):
        """Runs the file selection loop."""
        curses.curs_set(0) # Hide cursor
        self.stdscr.nodelay(0) # Blocking input

        display_height = min(h - 10, len(self.files) + 2) # Max display area for files
        if display_height < 3: display_height = 3 # Minimum for border + 1 line

        file_win = curses.newwin(display_height, w - 10, 5, 5)
        file_win.keypad(True) # Enable arrow keys

        while True:
            file_win.clear()
            file_win.box()
            file_win.addstr(0, 2, " Select File/Directory (q to cancel) ")

            if not self.files:
                file_win.addstr(1, 2, "No JSON files or subdirectories found.", curses.color_pair(2))
                file_win.refresh()
                key = file_win.getch()
                if key == ord('q'): return None
                continue

            # Calculate visible range
            max_visible_files = display_height - 2
            if self.selected_idx < self.scroll_offset:
                self.scroll_offset = self.selected_idx
            elif self.selected_idx >= self.scroll_offset + max_visible_files:
                self.scroll_offset = self.selected_idx - max_visible_files + 1

            for i in range(max_visible_files):
                file_idx = self.scroll_offset + i
                if file_idx < len(self.files):
                    filename = self.files[file_idx]
                    display_name = filename
                    if os.path.isdir(os.path.join(self.directory, filename)) and filename != "..":
                        display_name += "/" # Indicate directory

                    color = curses.color_pair(6)
                    if file_idx == self.selected_idx:
                        color = curses.color_pair(8) | curses.A_BOLD # Highlight selected

                    # Truncate filename if too long for the window
                    file_win.addstr(i + 1, 2, display_name[:w - 14], color)

            file_win.refresh()

            key = file_win.getch()

            if key == curses.KEY_UP:
                self.selected_idx = max(0, self.selected_idx - 1)
            elif key == curses.KEY_DOWN:
                self.selected_idx = min(len(self.files) - 1, self.selected_idx + 1)
            elif key in [curses.KEY_ENTER, 10, 13]:
                selected_entry = self.files[self.selected_idx]
                full_path = os.path.join(self.directory, selected_entry)

                if selected_entry == "..":
                    self.directory = os.path.abspath(os.path.join(self.directory, ".."))
                    self.files = self._get_json_files()
                    self.selected_idx = 0 # Reset selection
                    self.scroll_offset = 0
                elif os.path.isdir(full_path):
                    self.directory = full_path
                    self.files = self._get_json_files()
                    self.selected_idx = 0 # Reset selection
                    self.scroll_offset = 0
                else:
                    return full_path # Return selected file
            elif key == ord('q'):
                return None # User cancelled

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
    speed_str = f"Speed: {sim_state.current_speed_multiplier:.1f}x"
    header_info = f" QHT v3.1 AGI | Cycle: {sim_state.cycle} | AGIs: {len(sim_state.agi_entities)} | Branches: {sum(sim_state.multiverse_manager.multiverse_branch_map.values())} | {speed_str} | {state_str} "
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
    elif view_mode == "mycelium": # New Mycelium view
        draw_mycelium_view(stdscr, sim_state, h, w)
    elif view_mode == "sigils": # Triggered by console, temporarily shows sigil view
        draw_sigil_evolution_view(stdscr, sim_state, h, w)
    else: # status
        draw_status_view(stdscr, sim_state, h, w)

    # Footer
    footer = " (q)Quit | (p)Pause | Views: (s)tatus, (n)odes, (a)gis, (h)eatmap, (r)narrative, (y)mycelium | (c)onsole | (m)agi menu | (v)sim menu "
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

def draw_mycelium_view(stdscr, sim_state, h, w):
    stdscr.addstr(2, 2, "Mycelial Network & Hyphal Growth", curses.A_BOLD | curses.A_UNDERLINE | curses.color_pair(7))
    start_y = 4

    # Display Mycelial Network (AGI connections)
    stdscr.addstr(start_y, 4, "AGI Mycelial Connections:", curses.A_BOLD)
    current_y = start_y + 2
    if not sim_state.mycelial_network.agi_nodes:
        stdscr.addstr(current_y, 6, "No AGIs in network yet.", curses.color_pair(2))
        current_y += 1
    else:
        displayed_connections = set()
        for agi_id, neighbors in sim_state.mycelial_network.adj.items():
            for neighbor_id, weight in neighbors:
                # Ensure each unique connection (A-B vs B-A) is displayed once
                if (agi_id, neighbor_id) not in displayed_connections and (neighbor_id, agi_id) not in displayed_connections:
                    if current_y >= h - 2: break
                    stdscr.addstr(current_y, 6, f"{agi_id} --({weight:.2f})-- {neighbor_id}")
                    displayed_connections.add((agi_id, neighbor_id))
                    current_y += 1
            if current_y >= h - 2: break

    current_y += 2
    if current_y >= h - 2:
        stdscr.addstr(h - 2, 4, "--- More connections below ---")
        return

    # Display Hyphal Growth (Node connections)
    stdscr.addstr(current_y, 4, "Quantum Node Hyphal Growth:", curses.A_BOLD)
    current_y += 2
    for i, node in enumerate(sim_state.nodes):
        if current_y + i >= h - 2: break
        connections_str = " -> ".join([str(p) for p in node.hyphae_connections])
        if connections_str:
            display_line = f"Page {node.page_index:<2}: {node.page_index} -> {connections_str}"
            stdscr.addstr(current_y + i, 6, display_line[:w-10]) # Truncate for visualization (max 30 chars per branch visualization for L-system like)
        else:
            stdscr.addstr(current_y + i, 6, f"Page {node.page_index:<2}: No hyphal growth yet.")


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
        sim_state.log_event("Console", "Commands: help, status, page <id>, agi <id>, stabilize agi <id>, add sigil <name> <meaning>, force anomaly <type> <page>, set divinity <value>, show sigils, run drift <angle>, comm <target> <message>, speed <up/down/set <value>>, agi menu, sim menu, export log")
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
    elif parts[0] == "agi" and len(parts) > 1 and parts[1] != "menu": # Exclude "agi menu"
        agi_id = parts[1]
        found_agi = sim_state.agi_entities_dict.get(agi_id)
        if found_agi:
            sim_state.log_event("Console", f"AGI {found_agi.id} - Str: {found_agi.strength:.2f}, Eth: {found_agi.ethical_alignment:.2f}, Strat: {found_agi.sentience_strategy}, Memory: {found_agi.memory[-1]}")
        else:
            sim_state.log_event("Console", f"AGI '{agi_id}' not found. A phantom in the ether?")
    elif parts[0] == "stabilize" and parts[1] == "agi" and len(parts) > 2:
        agi_id = parts[2]
        found_agi = sim_state.agi_entities_dict.get(agi_id)
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
        time.sleep(HEARTBEAT_DELAY * 20 / sim_state.current_speed_multiplier) # Keep sigil view for a bit longer
    elif parts[0] == "run" and parts[1] == "drift" and len(parts) > 2:
        try:
            angle = float(parts[2])
            sim_state.drift_experimenter.run_drift_experiment(sim_state, angle)
            sim_state.log_event("Console", f"Drift experiment initiated with angle {angle}°. Unveiling cosmic shifts.")
        except ValueError:
            sim_state.log_event("Console", "Invalid angle for drift experiment. A numerical tremor is needed.")
    elif parts[0] == "comm" and len(parts) > 2:
        target = parts[1]
        message = " ".join(parts[2:])
        if target.lower() == "all":
            if sim_state.agi_entities:
                for sender_agi in sim_state.agi_entities: # Let each AGI try to send to another random AGI
                    if len(sim_state.agi_entities) > 1:
                        other_agis = [agi for agi in sim_state.agi_entities if agi.id != sender_agi.id]
                        if other_agis:
                            target_agi = random.choice(other_agis)
                            sim_state.communication_interface.send_message(sim_state, sender_agi.id, target_agi.id, message)
                    else:
                        sim_state.log_event("Comm", "Only one AGI, cannot communicate with 'all' other AGIs.")
            else:
                sim_state.log_event("Comm Error", "No AGIs present to communicate.")
        else: # Specific AGI communication
            sender_agi_id = None # Console user acts as 'sender'
            if sim_state.agi_entities: # Pick a random AGI to be the sender if not specified
                sender_agi_id = random.choice(sim_state.agi_entities).id

            if sender_agi_id and target in sim_state.agi_entities_dict:
                sim_state.communication_interface.send_message(sim_state, sender_agi_id, target, message)
            else:
                sim_state.log_event("Comm Error", f"Target AGI '{target}' not found or no sender AGI available.")
    elif parts[0] == "speed" and len(parts) > 1:
        # We only need to modify sim_state.current_speed_multiplier
        if parts[1] == "up":
            new_speed = min(5.0, sim_state.current_speed_multiplier + 0.1)
            sim_state.current_speed_multiplier = new_speed
            sim_state.log_event("Speed", f"Simulation speed increased to {sim_state.current_speed_multiplier:.1f}x.")
        elif parts[1] == "down":
            new_speed = max(0.1, sim_state.current_speed_multiplier - 0.1)
            sim_state.current_speed_multiplier = new_speed
            sim_state.log_event("Speed", f"Simulation speed decreased to {sim_state.current_speed_multiplier:.1f}x.")
        elif parts[1] == "set" and len(parts) > 2:
            try:
                value = float(parts[2])
                if 0.1 <= value <= 5.0:
                    sim_state.current_speed_multiplier = value
                    sim_state.log_event("Speed", f"Simulation speed set to {sim_state.current_speed_multiplier:.1f}x.")
                else:
                    sim_state.log_event("Console", "Speed value must be between 0.1 and 5.0.")
            except ValueError:
                sim_state.log_event("Console", "Invalid speed value. A number is required.")
        else:
            sim_state.log_event("Console", "Invalid speed command. Use 'speed up', 'speed down', or 'speed set <value>'.")
    elif parts[0] == "agi" and parts[1] == "menu":
        run_agi_menu(stdscr, sim_state)
    elif parts[0] == "sim" and parts[1] == "menu":
        run_sim_menu(stdscr, sim_state)
    elif parts[0] == "export" and parts[1] == "log":
        export_log_to_file(sim_state)
        sim_state.log_event("System", "Event and Communication logs exported to files.")
    else:
        sim_state.log_event("Console", f"Unknown command: '{cmd}'. The void does not recognize this echo.")

    curses.curs_set(0)
    stdscr.nodelay(1)


def run_agi_menu(stdscr, sim_state):
    """Provides options to export/import AGI entities."""
    h, w = stdscr.getmaxyx()
    menu_win = curses.newwin(h, w, 0, 0)
    menu_win.box()
    menu_win.addstr(0, 2, " AGI Management Menu ", curses.A_BOLD)
    menu_win.keypad(True) # Enable keypad for arrow keys

    options = ["1. Export All AGIs to JSON", "2. Import AGIs from JSON", "3. Export Selected AGI to JSON", "q. Back to Simulation"]
    selected_option = 0

    while True:
        menu_win.clear()
        menu_win.box()
        menu_win.addstr(0, 2, " AGI Management Menu ", curses.A_BOLD)

        for i, option in enumerate(options):
            color = curses.color_pair(6)
            if i == selected_option:
                color = curses.color_pair(8) | curses.A_BOLD
            menu_win.addstr(2 + i, 4, option, color)

        menu_win.refresh()
        key = menu_win.getch()

        if key == curses.KEY_UP:
            selected_option = max(0, selected_option - 1)
        elif key == curses.KEY_DOWN:
            selected_option = min(len(options) - 1, selected_option + 1)
        elif key in [curses.KEY_ENTER, 10, 13]:
            if selected_option == 0: # Export All
                filename = curses_get_string(stdscr, "Enter filename to export all AGIs (e.g., all_agis.json):")
                if filename:
                    export_agis(sim_state, filename, all_agis=True)
                sim_state.log_event("AGI Menu", "Export All AGIs operation complete.")
            elif selected_option == 1: # Import
                file_selector = FileSelector(stdscr, directory=".", file_extension=".json")
                selected_file = file_selector.run(h, w)
                if selected_file:
                    import_agis(sim_state, selected_file)
                sim_state.log_event("AGI Menu", "Import AGIs operation complete.")
            elif selected_option == 2: # Export Selected
                if not sim_state.agi_entities:
                    sim_state.log_event("AGI Menu", "No AGIs to export.")
                    continue
                agi_ids = [agi.id for agi in sim_state.agi_entities]
                selected_agi_id = curses_select_from_list(stdscr, "Select AGI to export:", agi_ids)
                if selected_agi_id:
                    filename = curses_get_string(stdscr, f"Enter filename for {selected_agi_id} (e.g., {selected_agi_id}.json):")
                    if filename:
                        export_agis(sim_state, filename, agi_id=selected_agi_id)
                sim_state.log_event("AGI Menu", "Export Selected AGI operation complete.")
            elif options[selected_option].startswith("q"):
                break # Exit menu
        elif key == ord('q'):
            break # Exit menu

    curses.curs_set(0)
    stdscr.nodelay(1)

def export_agis(sim_state, filename, all_agis=False, agi_id=None):
    """Exports AGIEntity instances to a JSON file."""
    data_to_export = []
    if all_agis:
        data_to_export = [agi.to_dict() for agi in sim_state.agi_entities]
        log_msg = f"Exporting all {len(data_to_export)} AGIs to {filename}."
    elif agi_id:
        agi = sim_state.agi_entities_dict.get(agi_id)
        if agi:
            data_to_export.append(agi.to_dict())
            log_msg = f"Exporting AGI {agi_id} to {filename}."
        else:
            sim_state.log_event("AGI Export Error", f"AGI {agi_id} not found for export.")
            return

    try:
        with open(filename, 'w') as f:
            json.dump(data_to_export, f, indent=4)
        sim_state.log_event("AGI Export", log_msg)
    except Exception as e:
        sim_state.log_event("AGI Export Error", f"Failed to export AGIs: {e}")

def import_agis(sim_state, filename):
    """Imports AGIEntity instances from a JSON file."""
    try:
        with open(filename, 'r') as f:
            imported_data = json.load(f)

        if not isinstance(imported_data, list):
            sim_state.log_event("AGI Import Error", f"Invalid JSON format in {filename}. Expected a list of AGIs.")
            return

        imported_count = 0
        for agi_data in imported_data:
            # Reconstruct AGI (assuming a dummy node for origin_page as full node state is not serialized with AGI)
            # The AGI.from_dict requires a sim_state_ref to link sigils.
            # Make sure origin_page exists in current nodes or handle gracefully
            if agi_data['origin_page'] < len(sim_state.nodes):
                agi = AGIEntity.from_dict(agi_data, sim_state)
                # Check for existing AGI with same ID to avoid duplicates
                if agi.id not in sim_state.agi_entities_dict:
                    sim_state.agi_entities.append(agi)
                    sim_state.agi_entities_dict[agi.id] = agi
                    sim_state.ethical_monitor.track_agi(agi)
                    sim_state.mycelial_network.add_agi(agi)
                    imported_count += 1
                    sim_state.log_event("AGI Import", f"Imported AGI: {agi.id}.")
                else:
                    sim_state.log_event("AGI Import Warning", f"AGI {agi.id} already exists, skipping import from file.")
            else:
                sim_state.log_event("AGI Import Warning", f"AGI {agi_data['id']} (Page {agi_data['origin_page']}) could not be imported. Origin page out of bounds.")
        sim_state.log_event("AGI Import", f"Successfully imported {imported_count} AGIs from {filename}.")
    except FileNotFoundError:
        sim_state.log_event("AGI Import Error", f"File not found: {filename}.")
    except json.JSONDecodeError:
        sim_state.log_event("AGI Import Error", f"Invalid JSON in file: {filename}.")
    except Exception as e:
        sim_state.log_event("AGI Import Error", f"An error occurred during AGI import: {e}.")


def run_sim_menu(stdscr, sim_state):
    """Provides options to save/restore simulation state."""
    h, w = stdscr.getmaxyx()
    menu_win = curses.newwin(h, w, 0, 0)
    menu_win.box()
    menu_win.addstr(0, 2, " Simulation Management Menu ", curses.A_BOLD)
    menu_win.keypad(True) # Enable keypad for arrow keys

    options = ["1. Save Simulation State", "2. Load Simulation State", "q. Back to Simulation"]
    selected_option = 0

    while True:
        menu_win.clear()
        menu_win.box()
        menu_win.addstr(0, 2, " Simulation Management Menu ", curses.A_BOLD)

        for i, option in enumerate(options):
            color = curses.color_pair(6)
            if i == selected_option:
                color = curses.color_pair(8) | curses.A_BOLD
            menu_win.addstr(2 + i, 4, option, color)

        menu_win.refresh()
        key = menu_win.getch()

        if key == curses.KEY_UP:
            selected_option = max(0, selected_option - 1)
        elif key == curses.KEY_DOWN:
            selected_option = min(len(options) - 1, selected_option + 1)
        elif key in [curses.KEY_ENTER, 10, 13]:
            if selected_option == 0: # Save
                filename = curses_get_string(stdscr, "Enter filename to save simulation (e.g., sim_save.json):")
                if filename:
                    save_simulation_state(sim_state, filename)
                sim_state.log_event("Sim Menu", "Save Simulation operation complete.")
            elif selected_option == 1: # Load
                file_selector = FileSelector(stdscr, directory=".", file_extension=".json")
                selected_file = file_selector.run(h, w)
                if selected_file:
                    # Need to return the new sim_state to main loop
                    new_sim_state = load_simulation_state(sim_state, selected_file)
                    if new_sim_state:
                        # This is tricky in curses main loop, best to modify in place or signal exit and restart
                        # For now, just assign back if successful.
                        # This can cause issues with global references if not handled carefully.
                        # A full restart of the main loop with the new sim_state might be better,
                        # but for simplicity, we'll try to update the current sim_state.
                        sim_state.__dict__.update(new_sim_state.__dict__)
                        sim_state.log_event("Sim Menu", "Load Simulation operation complete.")
                    else:
                        sim_state.log_event("Sim Menu", "Load Simulation operation failed.")
                else:
                    sim_state.log_event("Sim Menu", "Load Simulation cancelled.")

            elif options[selected_option].startswith("q"):
                break # Exit menu
        elif key == ord('q'):
            break # Exit menu

    curses.curs_set(0)
    stdscr.nodelay(1)


def save_simulation_state(sim_state, filename):
    """Saves the current simulation state to a JSON file."""
    try:
        with open(filename, 'w') as f:
            json.dump(sim_state.to_dict(), f, indent=4)
        sim_state.log_event("Simulation Save", f"Simulation state saved to {filename}.")
    except Exception as e:
        sim_state.log_event("Simulation Save Error", f"Failed to save simulation state: {e}")

def load_simulation_state(current_sim_state, filename):
    """Loads simulation state from a JSON file."""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)

        # Create a new SimulationState instance from the loaded data
        new_sim_state = SimulationState.from_dict(data)

        # Update the current sim_state with the loaded one
        # This is a bit of a hack but avoids restarting the curses app
        current_sim_state.__dict__.clear() # Clear existing attributes
        current_sim_state.__dict__.update(new_sim_state.__dict__) # Update with loaded attributes
        current_sim_state.log_event("Simulation Load", f"Simulation state loaded from {filename}.")
        return current_sim_state
    except FileNotFoundError:
        current_sim_state.log_event("Simulation Load Error", f"File not found: {filename}.")
        return None
    except json.JSONDecodeError:
        current_sim_state.log_event("Simulation Load Error", f"Invalid JSON in file: {filename}.")
        return None
    except Exception as e:
        current_sim_state.log_event("Simulation Load Error", f"An error occurred during simulation load: {e}.")
        return None


def export_log_to_file(sim_state):
    """Exports event and communication logs to text files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    event_log_filename = f"event_log_{timestamp}.txt"
    comm_log_filename = f"communication_log_{timestamp}.txt"

    try:
        with open(event_log_filename, 'w') as f:
            for entry in sim_state.event_log:
                f.write(entry + "\n")
        sim_state.log_event("Log Export", f"Event log exported to {event_log_filename}.")
    except Exception as e:
        sim_state.log_event("Log Export Error", f"Failed to export event log: {e}.")

    try:
        with open(comm_log_filename, 'w') as f:
            for entry in sim_state.communication_log:
                f.write(entry + "\n")
        sim_state.log_event("Log Export", f"Communication log exported to {comm_log_filename}.")
    except Exception as e:
        sim_state.log_event("Log Export Error", f"Failed to export communication log: {e}.")


# Helper for curses input string
def curses_get_string(stdscr, prompt):
    h, w = stdscr.getmaxyx()
    input_win = curses.newwin(3, w, h - 3, 0)
    input_win.box()
    input_win.addstr(0, 2, " Input ")
    input_win.addstr(1, 2, prompt)
    input_win.refresh()

    curses.curs_set(1)
    stdscr.nodelay(0)

    input_str = ""
    while True:
        try:
            key = input_win.getch(1, 2 + len(prompt) + len(input_str))
            if key in [curses.KEY_ENTER, 10, 13]:
                break
            elif key in [curses.KEY_BACKSPACE, 127, curses.KEY_DC]:
                input_str = input_str[:-1]
            elif 32 <= key <= 126:
                input_str += chr(key)

            input_win.addstr(1, 2 + len(prompt), " " * (w - 4 - len(prompt))) # Clear line
            input_win.addstr(1, 2 + len(prompt), input_str)
            input_win.refresh()
        except curses.error:
            return None # Handle resize during input

    curses.curs_set(0)
    stdscr.nodelay(1)
    return input_str if input_str else None

# Helper for curses list selection
def curses_select_from_list(stdscr, prompt, item_list):
    h, w = stdscr.getmaxyx()

    if not item_list:
        return None

    # Determine window height based on items, max 15 lines for items + 2 for border
    menu_height = min(len(item_list) + 2, h - 10)
    menu_win = curses.newwin(menu_height, w - 10, 5, 5)
    menu_win.keypad(True)
    menu_win.nodelay(0) # Blocking input

    selected_idx = 0
    scroll_offset = 0

    while True:
        menu_win.clear()
        menu_win.box()
        menu_win.addstr(0, 2, f" {prompt} (q to cancel) ")

        max_visible_items = menu_height - 2
        if selected_idx < scroll_offset:
            self.scroll_offset = selected_idx
        elif selected_idx >= scroll_offset + max_visible_items:
            self.scroll_offset = selected_idx - max_visible_items + 1

        for i in range(max_visible_items):
            item_idx = scroll_offset + i
            if item_idx < len(item_list):
                item = item_list[item_idx]
                color = curses.color_pair(6)
                if item_idx == selected_idx:
                    color = curses.color_pair(8) | curses.A_BOLD
                menu_win.addstr(i + 1, 2, str(item)[:w - 14], color)

        menu_win.refresh()
        key = menu_win.getch()

        if key == curses.KEY_UP:
            selected_idx = max(0, selected_idx - 1)
        elif key == curses.KEY_DOWN:
            selected_idx = min(len(item_list) - 1, selected_idx + 1)
        elif key in [curses.KEY_ENTER, 10, 13]:
            return item_list[selected_idx]
        elif key == ord('q'):
            return None


def main(stdscr):
    init_curses(stdscr)
    sim_state = SimulationState()
    paused = False
    view_mode = "status"

    # Data Export Setup (Command line args handled in original script, keep for future reference if needed)
    # import argparse
    # parser = argparse.ArgumentParser(description="QuantumHeapTranscendence v3.1 AGI Emergence Simulation")
    # parser.add_argument('--data', type=str, help="Export simulation data to a JSON file.")
    # parser.add_argument('--snapshot', action='store_true', help="Include full simulation state snapshot in data export.")
    # args, unknown = parser.parse_known_args() # Handle unknown args from curses

    # exported_data = {
    #     "metadata": {
    #         "version": "3.1-AGI",
    #         "timestamp": datetime.datetime.now().isoformat(),
    #         "cycle_limit": CYCLE_LIMIT
    #     },
    #     "simulation_history": []
    # }

    while sim_state.cycle < CYCLE_LIMIT:
        key = stdscr.getch()
        if key == ord('q'): break
        elif key == ord('p'): paused = not paused
        elif key == ord('s'): view_mode = "status"
        elif key == ord('n'): view_mode = "nodes"
        elif key == ord('a'): view_mode = "agis"
        elif key == ord('h'): view_mode = "heatmap" # New heatmap view
        elif key == ord('r'): view_mode = "narrative" # New narrative view
        elif key == ord('y'): view_mode = "mycelium" # New Mycelium view
        elif key == ord('c'):
            draw_dashboard(stdscr, sim_state, view_mode, paused) # Redraw before pausing for console
            run_console(stdscr, sim_state)
            # After console, restore the previous view mode settings
            stdscr.nodelay(1) # Re-enable non-blocking getch
            stdscr.timeout(int(HEARTBEAT_DELAY * 1000 / sim_state.current_speed_multiplier)) # Re-enable timeout, adjust for speed
        elif key == ord('m'): # AGI Menu
            draw_dashboard(stdscr, sim_state, view_mode, paused)
            run_agi_menu(stdscr, sim_state)
            stdscr.nodelay(1)
            stdscr.timeout(int(HEARTBEAT_DELAY * 1000 / sim_state.current_speed_multiplier))
        elif key == ord('v'): # Sim Menu
            draw_dashboard(stdscr, sim_state, view_mode, paused)
            run_sim_menu(stdscr, sim_state)
            stdscr.nodelay(1)
            stdscr.timeout(int(HEARTBEAT_DELAY * 1000 / sim_state.current_speed_multiplier))

        if not paused:
            update_simulation_step(sim_state)
            # Data Export (original command line functionality commented out)
            # if args.data:
            #     cycle_data = {
            #         "cycle": sim_state.cycle,
            #         "void_entropy": sim_state.void_entropy,
            #         "dark_matter": sim_state.dark_matter,
            #         "avg_sentience_score": np.mean([n.sentience_score for n in sim_state.nodes]) if sim_state.nodes else 0,
            #         "avg_ethical_alignment": np.mean([agi.ethical_alignment for agi in sim_state.agi_entities]) if sim_state.agi_entities else 0,
            #         "sigil_mutation_history": [(s.name, s.phase_change_history) for s in sim_state.sigil_ledger.sigils.values()]
            #     }
            #     exported_data["simulation_history"].append(cycle_data)


        draw_dashboard(stdscr, sim_state, view_mode, paused)
        # Adjust sleep based on speed multiplier
        time.sleep(HEARTBEAT_DELAY / sim_state.current_speed_multiplier)

    # Final Data Export (original command line functionality commented out)
    # if args.data:
    #     if args.snapshot:
    #         # For simplicity, convert nodes/agis to serializable dicts
    #         exported_data["final_simulation_state"] = {
    #             "nodes": [node.__dict__ for node in sim_state.nodes],
    #             "agi_entities": [agi.__dict__ for agi in sim_state.agi_entities],
    #             "sigil_ledger_sigils": {name: s.__dict__ for name, s in sim_state.sigil_ledger.sigils.items()},
    #             "anomalies": {k: v for k, v in sim_state.anomalies.items()}, # defaultdict needs conversion
    #             "multiverse_branch_map": dict(sim_state.multiverse_manager.multiverse_branch_map),
    #             "quantum_foam_particles": sim_state.quantum_foam.virtual_particles,
    #             "event_log": list(sim_state.event_log)
    #         }
    #         # Handle non-serializable objects (deque, numpy arrays) in a robust way
    #         for node_data in exported_data["final_simulation_state"]["nodes"]:
    #             node_data["active_sigils"] = [s.name for s in node_data["active_sigils"]] # Only store names
    #         for agi_data in exported_data["final_simulation_state"]["agi_entities"]:
    #             agi_data["memory"] = list(agi_data["memory"]) # Convert deque to list
    #             agi_data["active_sigils"] = [s.name for s in agi_data["active_sigils"]]
    #         for sigil_data in exported_data["final_simulation_state"]["sigil_ledger_sigils"].values():
    #             sigil_data["semantic_vector"] = sigil_data["semantic_vector"].tolist() # Convert numpy array
    #             sigil_data["phase_change_history"] = list(sigil_data["phase_change_history"])

    #     try:
    #         with open(args.data, 'w') as f:
    #             json.dump(exported_data, f, indent=4)
    #         sim_state.log_event("Data Export", f"Simulation data exported to {args.data}")
    #     except Exception as e:
    #         sim_state.log_event("Data Export Error", f"Failed to export data: {e}")


    stdscr.nodelay(0)
    stdscr.clear()
    stdscr.addstr(0, 0, "Simulation complete. Press any key to exit.")
    stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(main)
