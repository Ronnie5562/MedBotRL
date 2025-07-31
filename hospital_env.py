import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
from typing import Dict, Tuple, List, Optional
from enum import Enum

class PatientUrgency(Enum):
    LOW = 1
    MODERATE = 2
    CRITICAL = 3

class RoomType(Enum):
    EMERGENCY = 1
    ICU = 2
    SURGERY = 3
    GENERAL = 4
    LAB = 5
    PHARMACY = 6
    RADIOLOGY = 7
    CARDIOLOGY = 8
    PEDIATRICS = 9
    NEUROLOGY = 10

class HospitalNavigationEnv(gym.Env):
    """
    Hospital Navigation Environment for Telemedicine Platform - UPDATED VERSION

    The agent must navigate through hospital corridors to:
    1. Collect patients from corridors
    2. Get drugs from pharmacy when patients need medication
    3. Deliver patients to appropriate departments
    4. Maximize efficiency and patient care

    IMPROVEMENTS:
    - More flexible movement system
    - Better reward shaping with distance-based rewards
    - Reduced harsh penalties
    - Progressive difficulty options
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None, max_steps=1000, simple_mode=False):
        super().__init__()

        # Environment dimensions
        self.WORLD_WIDTH = 1000
        self.WORLD_HEIGHT = 700
        self.CORRIDOR_WIDTH = 40

        # Agent properties
        self.AGENT_SIZE = 25
        self.PATIENT_SIZE = 20
        self.DRUG_STATION_SIZE = 15
        self.MOVEMENT_SPEED = 15  # Reduced from 20 for smoother movement

        # Simple mode for easier learning
        self.simple_mode = simple_mode

        # Define corridor grid (these are the preferred positions but not mandatory)
        self.corridor_positions = self._generate_corridor_positions()

        # Define room boundaries (agent cannot enter these)
        self.room_boundaries = self._define_room_boundaries()

        # Define drug stations (near pharmacy) - FIXED POSITIONS with more stations
        # Make sure they're in valid corridor positions and increase availability
        self.drug_stations = [(770, 420), (770, 440), (270, 420), (520, 420)]  # 4 drug stations for better balance

        # Action space: 8 directional movements + stay still
        # 0: Up, 1: Down, 2: Left, 3: Right, 4: Up-Left, 5: Up-Right, 6: Down-Left, 7: Down-Right, 8: Stay
        self.action_space = spaces.Discrete(9)

        # Observation space: Enhanced with more information
        obs_size = 2 + 2 + (6 * 4) + 4 + 4  # agent + status + patients + drugs + distances
        self.observation_space = spaces.Box(
            low=0, high=max(self.WORLD_WIDTH, self.WORLD_HEIGHT),
            shape=(obs_size,), dtype=np.float32
        )

        # Game state
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode

        # Tracking for better rewards
        self.previous_distances = {'patient': float('inf'), 'drug': float('inf'), 'room': float('inf')}
        self.steps_without_progress = 0

        # Initialize pygame for rendering
        if self.render_mode == 'human':
            pygame.init()
            self.screen = pygame.display.set_mode((self.WORLD_WIDTH, self.WORLD_HEIGHT))
            pygame.display.set_caption("Hospital Navigation RL Environment - Updated")
            self.clock = pygame.time.Clock()
            self.font_small = pygame.font.Font(None, 16)
            self.font_medium = pygame.font.Font(None, 20)
            self.font_large = pygame.font.Font(None, 24)

        self.reset()

    def _generate_corridor_positions(self) -> List[Tuple[int, int]]:
        """Generate corridor positions - now used as reference points"""
        positions = []

        # Main horizontal corridors
        corridor_y_positions = [20, 220, 420, 660]
        for y in corridor_y_positions:
            for x in range(20, self.WORLD_WIDTH - 20, 20):
                positions.append((x, y))

        # Main vertical corridors
        corridor_x_positions = [20, 270, 520, 770, 960]
        for x in corridor_x_positions:
            for y in range(60, self.WORLD_HEIGHT - 60, 20):
                if y not in [20, 220, 420, 660]:
                    positions.append((x, y))

        return positions

    def _define_room_boundaries(self) -> List[Dict]:
        """Define room boundaries that agent cannot enter"""
        return [
            # Row 1 Rooms
            {'x': 50, 'y': 50, 'width': 140, 'height': 100, 'type': RoomType.EMERGENCY, 'name': 'Emergency'},
            {'x': 300, 'y': 50, 'width': 140, 'height': 100, 'type': RoomType.EMERGENCY, 'name': 'Emergency 2'},
            {'x': 550, 'y': 50, 'width': 140, 'height': 100, 'type': RoomType.ICU, 'name': 'ICU'},
            {'x': 800, 'y': 50, 'width': 140, 'height': 100, 'type': RoomType.SURGERY, 'name': 'Surgery 1'},

            # Row 2 Rooms
            {'x': 50, 'y': 250, 'width': 140, 'height': 100, 'type': RoomType.GENERAL, 'name': 'General 1'},
            {'x': 300, 'y': 250, 'width': 140, 'height': 100, 'type': RoomType.CARDIOLOGY, 'name': 'Cardiology'},
            {'x': 550, 'y': 250, 'width': 140, 'height': 100, 'type': RoomType.NEUROLOGY, 'name': 'Neurology'},
            {'x': 800, 'y': 250, 'width': 140, 'height': 100, 'type': RoomType.SURGERY, 'name': 'Surgery 2'},

            # Row 3 Rooms
            {'x': 50, 'y': 450, 'width': 140, 'height': 100, 'type': RoomType.LAB, 'name': 'Laboratory'},
            {'x': 300, 'y': 450, 'width': 140, 'height': 100, 'type': RoomType.RADIOLOGY, 'name': 'Radiology'},
            {'x': 550, 'y': 450, 'width': 140, 'height': 100, 'type': RoomType.PEDIATRICS, 'name': 'Pediatrics'},
            {'x': 800, 'y': 450, 'width': 140, 'height': 100, 'type': RoomType.PHARMACY, 'name': 'Pharmacy'},

            # Row 4 Rooms (Wards)
            {'x': 50, 'y': 600, 'width': 140, 'height': 80, 'type': RoomType.GENERAL, 'name': 'Ward A'},
            {'x': 300, 'y': 600, 'width': 140, 'height': 80, 'type': RoomType.GENERAL, 'name': 'Ward B'},
            {'x': 550, 'y': 600, 'width': 140, 'height': 80, 'type': RoomType.GENERAL, 'name': 'Ward C'},
            {'x': 800, 'y': 600, 'width': 140, 'height': 80, 'type': RoomType.GENERAL, 'name': 'Ward D'},
        ]

    def _is_inside_room(self, x: int, y: int) -> bool:
        """Check if position is inside any room (forbidden area)"""
        for room in self.room_boundaries:
            if (room['x'] < x < room['x'] + room['width'] and
                room['y'] < y < room['y'] + room['height']):
                return True
        return False

    def _is_on_corridor(self, x: int, y: int) -> bool:
        """Check if position is on a valid corridor path"""
        # Define corridor zones more precisely
        corridor_zones = [
            # Horizontal corridors (with some width tolerance)
            {'x_min': 0, 'x_max': self.WORLD_WIDTH, 'y_min': 10, 'y_max': 30},    # Top corridor
            {'x_min': 0, 'x_max': self.WORLD_WIDTH, 'y_min': 210, 'y_max': 230}, # Second corridor
            {'x_min': 0, 'x_max': self.WORLD_WIDTH, 'y_min': 410, 'y_max': 430}, # Third corridor
            {'x_min': 0, 'x_max': self.WORLD_WIDTH, 'y_min': 650, 'y_max': 670}, # Bottom corridor

            # Vertical corridors (with some width tolerance)
            {'x_min': 10, 'x_max': 30, 'y_min': 0, 'y_max': self.WORLD_HEIGHT},    # Left corridor
            {'x_min': 260, 'x_max': 280, 'y_min': 0, 'y_max': self.WORLD_HEIGHT}, # Second corridor
            {'x_min': 510, 'x_max': 530, 'y_min': 0, 'y_max': self.WORLD_HEIGHT}, # Third corridor
            {'x_min': 760, 'x_max': 780, 'y_min': 0, 'y_max': self.WORLD_HEIGHT}, # Fourth corridor
            {'x_min': 950, 'x_max': 970, 'y_min': 0, 'y_max': self.WORLD_HEIGHT}, # Right corridor
        ]

        # Check if position is in any corridor zone
        for zone in corridor_zones:
            if (zone['x_min'] <= x <= zone['x_max'] and
                zone['y_min'] <= y <= zone['y_max']):
                return True
        return False

    def _is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is valid (in corridors, not inside rooms, within bounds)"""
        return (0 <= x < self.WORLD_WIDTH and
                0 <= y < self.WORLD_HEIGHT and
                not self._is_inside_room(x, y) and
                self._is_on_corridor(x, y))

    def _get_nearest_corridor_position(self, x: int, y: int) -> Tuple[int, int]:
        """Get nearest valid corridor position"""
        # First, try to find a valid position near the requested coordinates
        for radius in range(0, 100, 10):
            for angle in range(0, 360, 45):
                test_x = x + radius * math.cos(math.radians(angle))
                test_y = y + radius * math.sin(math.radians(angle))
                if self._is_valid_position(int(test_x), int(test_y)):
                    return (int(test_x), int(test_y))

        # Fallback to predefined corridor positions
        min_distance = float('inf')
        nearest_pos = self.corridor_positions[0]
        for corridor_x, corridor_y in self.corridor_positions:
            distance = math.sqrt((x - corridor_x)**2 + (y - corridor_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_pos = (corridor_x, corridor_y)
        return nearest_pos

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset agent to starting position
        self.agent_pos = (220, 220)  # Start in a central corridor position

        # Reset game state
        self.current_step = 0
        self.carrying_patient = None
        self.has_drugs = False
        self.score = 0
        self.patients_saved = 0
        self.total_reward = 0
        self.steps_without_progress = 0
        self.previous_distances = {'patient': float('inf'), 'drug': float('inf'), 'room': float('inf')}

        # Generate patients in corridor positions
        self.patients = []
        num_patients = 3 if self.simple_mode else 6
        self._spawn_patients(num_patients)

        # Drug station availability - more stations now
        self.drug_stations_available = [True, True, True, True]

        return self._get_observation(), {}

    def _spawn_patients(self, num_patients: int):
        """Spawn patients in random valid corridor positions with balanced drug requirements"""
        urgency_mapping = {
            PatientUrgency.CRITICAL: {'target_rooms': [RoomType.EMERGENCY, RoomType.ICU, RoomType.SURGERY]},
            PatientUrgency.MODERATE: {'target_rooms': [RoomType.CARDIOLOGY, RoomType.NEUROLOGY, RoomType.GENERAL]},
            PatientUrgency.LOW: {'target_rooms': [RoomType.PHARMACY, RoomType.LAB, RoomType.PEDIATRICS]}
        }

        # Calculate balanced drug requirements
        # Ensure at most 2 patients need drugs initially (matching available drug stations)
        max_drug_patients = min(2, num_patients)  # Never more than 2
        patients_needing_drugs = 0

        for i in range(num_patients):
            # Find random valid position (not necessarily on corridor grid)
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = random.randint(50, self.WORLD_WIDTH - 50)
                y = random.randint(50, self.WORLD_HEIGHT - 50)
                if self._is_valid_position(x, y):
                    pos = (x, y)
                    break
                attempts += 1
            else:
                # Fallback to corridor position
                pos = random.choice(self.corridor_positions)

            urgency = random.choice(list(PatientUrgency))
            target_room = random.choice(urgency_mapping[urgency]['target_rooms'])

            # Balanced drug requirement - only allow if we haven't reached the limit
            if patients_needing_drugs < max_drug_patients and random.random() < 0.6:
                needs_drugs = True
                patients_needing_drugs += 1
            else:
                needs_drugs = False

            patient = {
                'position': pos,
                'urgency': urgency,
                'target_room': target_room,
                'needs_drugs': needs_drugs,
                'treated_with_drugs': False
            }

            self.patients.append(patient)

        print(f"🏥 Spawned {num_patients} patients, {patients_needing_drugs} need drugs")

    def step(self, action):
        self.current_step += 1

        # Calculate movement
        movement_map = {
            0: (0, -self.MOVEMENT_SPEED),   # Up
            1: (0, self.MOVEMENT_SPEED),    # Down
            2: (-self.MOVEMENT_SPEED, 0),   # Left
            3: (self.MOVEMENT_SPEED, 0),    # Right
            4: (-self.MOVEMENT_SPEED, -self.MOVEMENT_SPEED), # Up-Left
            5: (self.MOVEMENT_SPEED, -self.MOVEMENT_SPEED),  # Up-Right
            6: (-self.MOVEMENT_SPEED, self.MOVEMENT_SPEED),  # Down-Left
            7: (self.MOVEMENT_SPEED, self.MOVEMENT_SPEED),   # Down-Right
            8: (0, 0)  # Stay still
        }

        dx, dy = movement_map[action]
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        # Check if new position is valid
        if self._is_valid_position(new_x, new_y):
            self.agent_pos = (new_x, new_y)

        # Calculate reward
        reward = self._calculate_reward(action)
        self.total_reward += reward

        # Check if episode is done
        done = (self.current_step >= self.max_steps or len(self.patients) == 0)

        return self._get_observation(), reward, done, False, {}

    def _calculate_reward(self, action) -> float:
        """Enhanced reward calculation with distance-based rewards and path compliance"""
        reward = -0.01  # Very small step penalty

        agent_x, agent_y = self.agent_pos

        # Progress tracking
        made_progress = False

        # PENALTY for being off corridor paths (encourages realistic movement)
        if not self._is_on_corridor(agent_x, agent_y):
            reward -= 5.0  # Strong penalty for leaving corridors
            return reward  # Return immediately to discourage this behavior

        # BONUS for staying on proper paths
        reward += 0.02  # Small bonus for following corridors

        # 1. Drug Station Interaction
        if not self.has_drugs:
            for i, (drug_x, drug_y) in enumerate(self.drug_stations):
                distance_to_drug = math.sqrt((agent_x - drug_x)**2 + (agent_y - drug_y)**2)

                # Distance-based reward for approaching drug station (FIXED: prevent division by zero)
                if distance_to_drug < self.previous_distances['drug'] and self.previous_distances['drug'] != float('inf'):
                    reward_delta = max(0, min(10, (self.previous_distances['drug'] - distance_to_drug) / 10))
                    reward += reward_delta
                    made_progress = True

                # Collection reward
                if (self.drug_stations_available[i] and distance_to_drug < 25):
                    self.has_drugs = True
                    self.drug_stations_available[i] = False
                    reward += 20
                    made_progress = True
                    print(f"🏥 Collected drugs from station {i}!")

                    # Respawn drug station after some time (increased chance)
                    if random.random() < 0.3:  # Increased from 0.1 to 0.3
                        self.drug_stations_available[i] = True

                # Update distance tracking
                if distance_to_drug < 500:  # Only track if reasonably close
                    self.previous_distances['drug'] = min(self.previous_distances['drug'], distance_to_drug)

        # 2. Patient Interactions
        patients_to_remove = []
        for i, patient in enumerate(self.patients):
            patient_x, patient_y = patient['position']
            distance_to_patient = math.sqrt((agent_x - patient_x)**2 + (agent_y - patient_y)**2)

            # Distance-based reward for approaching patients (FIXED: prevent division by zero)
            if distance_to_patient < self.previous_distances['patient'] and self.previous_distances['patient'] != float('inf'):
                reward_delta = max(0, min(5, (self.previous_distances['patient'] - distance_to_patient) / 20))
                reward += reward_delta
                made_progress = True

            # Patient interaction
            if distance_to_patient < 30:
                if patient['needs_drugs'] and self.has_drugs and not patient['treated_with_drugs']:
                    # Deliver drugs to patient
                    patient['treated_with_drugs'] = True
                    patient['needs_drugs'] = False
                    self.has_drugs = False
                    reward += 40
                    made_progress = True
                    print(f"💊 Delivered drugs to {patient['urgency'].name} patient!")

                elif not patient['needs_drugs'] and self.carrying_patient is None:
                    # Pick up patient
                    self.carrying_patient = patient
                    patients_to_remove.append(i)
                    reward += 25
                    made_progress = True
                    print(f"🚑 Picked up {patient['urgency'].name} patient!")

            # Update distance tracking
            if distance_to_patient < 500:  # Only track if reasonably close
                self.previous_distances['patient'] = min(self.previous_distances['patient'], distance_to_patient)

        # Remove picked up patients
        for i in reversed(patients_to_remove):
            del self.patients[i]

        # 3. Room Delivery
        if self.carrying_patient is not None:
            target_room_type = self.carrying_patient['target_room']
            min_room_distance = float('inf')

            for room in self.room_boundaries:
                if room['type'] == target_room_type:
                    room_center_x = room['x'] + room['width'] // 2
                    room_center_y = room['y'] + room['height'] // 2
                    room_distance = math.sqrt((agent_x - room_center_x)**2 + (agent_y - room_center_y)**2)
                    min_room_distance = min(min_room_distance, room_distance)

                    # Distance-based reward for approaching target room (FIXED: prevent division by zero)
                    if (room_distance < self.previous_distances['room'] and
                        self.previous_distances['room'] != float('inf') and
                        room_distance > 0):
                        reward_delta = max(0, min(8, (self.previous_distances['room'] - room_distance) / 15))
                        reward += reward_delta
                        made_progress = True

                    # Delivery reward - allow delivery from corridor adjacent to room
                    if room_distance < 100:  # Increased range for corridor-to-room delivery
                        urgency_rewards = {
                            PatientUrgency.CRITICAL: 100,
                            PatientUrgency.MODERATE: 70,
                            PatientUrgency.LOW: 50
                        }
                        reward += urgency_rewards[self.carrying_patient['urgency']]
                        self.patients_saved += 1
                        print(f"✅ Delivered {self.carrying_patient['urgency'].name} patient to {room['name']}!")
                        self.carrying_patient = None
                        made_progress = True
                        self.previous_distances['room'] = float('inf')  # Reset room distance
                        break

            if self.carrying_patient is not None and min_room_distance < 500:  # Still carrying
                self.previous_distances['room'] = min(self.previous_distances['room'], min_room_distance)

        # 4. Progress tracking and penalties
        if made_progress:
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        # Penalty for no progress (but much gentler)
        if self.steps_without_progress > 50:
            reward -= 0.1

        # Small penalty for staying still when there's work to do
        if action == 8 and (self.patients or self.carrying_patient):
            reward -= 0.05

        # Bonus for efficiency
        if len(self.patients) == 0:
            efficiency_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps) * 50
            reward += efficiency_bonus

        # Clamp reward to prevent infinity
        reward = max(-100, min(100, reward))

        return reward

    def _get_observation(self) -> np.ndarray:
        """Enhanced observation with distance information"""
        obs = []

        # Agent position (normalized)
        obs.extend([self.agent_pos[0] / self.WORLD_WIDTH, self.agent_pos[1] / self.WORLD_HEIGHT])

        # Agent status
        obs.append(1.0 if self.carrying_patient else 0.0)
        obs.append(1.0 if self.has_drugs else 0.0)

        # Patient information (pad to 6 patients)
        for i in range(6):
            if i < len(self.patients):
                patient = self.patients[i]
                obs.extend([
                    patient['position'][0] / self.WORLD_WIDTH,
                    patient['position'][1] / self.WORLD_HEIGHT,
                    patient['urgency'].value / 3.0,
                    1.0 if patient['needs_drugs'] else 0.0
                ])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        # Drug station availability (updated size)
        obs.extend([1.0 if available else 0.0 for available in self.drug_stations_available[:4]])  # Ensure 4 stations

        # Distance information (normalized)
        nearest_patient_dist = float('inf')
        nearest_drug_dist = float('inf')
        nearest_target_room_dist = float('inf')

        # Find nearest patient
        if self.patients and not self.carrying_patient:
            for patient in self.patients:
                dist = math.sqrt((self.agent_pos[0] - patient['position'][0])**2 +
                               (self.agent_pos[1] - patient['position'][1])**2)
                nearest_patient_dist = min(nearest_patient_dist, dist)

        # Find nearest drug station
        if not self.has_drugs:
            for i, (drug_x, drug_y) in enumerate(self.drug_stations):
                if self.drug_stations_available[i]:
                    dist = math.sqrt((self.agent_pos[0] - drug_x)**2 + (self.agent_pos[1] - drug_y)**2)
                    nearest_drug_dist = min(nearest_drug_dist, dist)

        # Find nearest target room
        if self.carrying_patient:
            target_room_type = self.carrying_patient['target_room']
            for room in self.room_boundaries:
                if room['type'] == target_room_type:
                    room_center_x = room['x'] + room['width'] // 2
                    room_center_y = room['y'] + room['height'] // 2
                    dist = math.sqrt((self.agent_pos[0] - room_center_x)**2 +
                                   (self.agent_pos[1] - room_center_y)**2)
                    nearest_target_room_dist = min(nearest_target_room_dist, dist)

        # Normalize distances
        max_distance = math.sqrt(self.WORLD_WIDTH**2 + self.WORLD_HEIGHT**2)
        obs.extend([
            min(nearest_patient_dist / max_distance, 1.0),
            min(nearest_drug_dist / max_distance, 1.0),
            min(nearest_target_room_dist / max_distance, 1.0),
            self.steps_without_progress / 100.0  # Progress indicator
        ])

        return np.array(obs, dtype=np.float32)

    def _draw_hud(self):
        """Draw enhanced HUD status panel"""
        hud_height = 140
        hud_surface = pygame.Surface((self.WORLD_WIDTH, hud_height))
        hud_surface.set_alpha(200)
        hud_surface.fill((20, 20, 30))
        self.screen.blit(hud_surface, (0, self.WORLD_HEIGHT - hud_height))

        pygame.draw.rect(self.screen, (255, 255, 255),
                        (0, self.WORLD_HEIGHT - hud_height, self.WORLD_WIDTH, hud_height), 2)

        y_offset = self.WORLD_HEIGHT - hud_height + 10

        # Left column - Basic info
        texts = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Total Reward: {self.total_reward:.1f}",
            f"Patients Saved: {self.patients_saved}",
            f"Remaining Patients: {len(self.patients)}",
            f"Progress Steps: {self.steps_without_progress}"
        ]

        for i, text in enumerate(texts):
            color = (255, 100, 100) if i == 4 and self.steps_without_progress > 30 else (255, 255, 255)
            text_surface = self.font_medium.render(text, True, color)
            self.screen.blit(text_surface, (10, y_offset + i * 20))

        # Middle column - Agent Status
        status_x = 300
        status_texts = [
            "Agent Status:",
            f"Carrying: {'Yes' if self.carrying_patient else 'No'}",
            f"Has Drugs: {'Yes' if self.has_drugs else 'No'}",
            f"Position: ({self.agent_pos[0]}, {self.agent_pos[1]})",
            f"Mode: {'Simple' if self.simple_mode else 'Full'}"
        ]

        for i, text in enumerate(status_texts):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            text_surface = self.font_medium.render(text, True, color)
            self.screen.blit(text_surface, (status_x, y_offset + i * 20))

        # Right column - Patient/Task Info
        task_x = 600
        if self.carrying_patient:
            task_texts = [
                "Current Task:",
                f"Patient: {self.carrying_patient['urgency'].name}",
                f"Target: {self.carrying_patient['target_room'].name}",
                f"Needs Drugs: {'Yes' if self.carrying_patient['needs_drugs'] else 'No'}",
                "→ Deliver to target room"
            ]
        elif not self.has_drugs and any(p['needs_drugs'] for p in self.patients):
            task_texts = [
                "Current Task:",
                "→ Get drugs from pharmacy",
                f"Patients needing drugs: {sum(1 for p in self.patients if p['needs_drugs'])}",
                "",
                ""
            ]
        elif self.patients:
            task_texts = [
                "Current Task:",
                "→ Pick up patient",
                f"Available patients: {len(self.patients)}",
                "",
                ""
            ]
        else:
            task_texts = [
                "Task Status:",
                "✓ All patients saved!",
                "",
                "",
                ""
            ]

        for i, text in enumerate(task_texts):
            if text:
                color = (100, 255, 100) if text.startswith('✓') else (255, 255, 255) if i == 0 else (200, 200, 200)
                text_surface = self.font_medium.render(text, True, color)
                self.screen.blit(text_surface, (task_x, y_offset + i * 20))

    def render(self):
        if self.render_mode == 'human':
            return self._render_human()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb_array()

    def _render_human(self):
        # Clear screen
        self.screen.fill((30, 30, 50))

        # Draw corridor areas with clear visual paths
        corridor_surface = pygame.Surface((self.WORLD_WIDTH, self.WORLD_HEIGHT))
        corridor_surface.set_alpha(60)

        # Draw actual corridor zones more clearly
        corridor_zones = [
            # Horizontal corridors
            {'x': 0, 'y': 10, 'width': self.WORLD_WIDTH, 'height': 20},
            {'x': 0, 'y': 210, 'width': self.WORLD_WIDTH, 'height': 20},
            {'x': 0, 'y': 410, 'width': self.WORLD_WIDTH, 'height': 20},
            {'x': 0, 'y': 650, 'width': self.WORLD_WIDTH, 'height': 20},

            # Vertical corridors
            {'x': 10, 'y': 0, 'width': 20, 'height': self.WORLD_HEIGHT},
            {'x': 260, 'y': 0, 'width': 20, 'height': self.WORLD_HEIGHT},
            {'x': 510, 'y': 0, 'width': 20, 'height': self.WORLD_HEIGHT},
            {'x': 760, 'y': 0, 'width': 20, 'height': self.WORLD_HEIGHT},
            {'x': 950, 'y': 0, 'width': 20, 'height': self.WORLD_HEIGHT},
        ]

        # Draw corridor paths
        for zone in corridor_zones:
            pygame.draw.rect(corridor_surface, (120, 120, 140),
                           (zone['x'], zone['y'], zone['width'], zone['height']))

        self.screen.blit(corridor_surface, (0, 0))

        # Draw corridor borders for clarity
        for zone in corridor_zones:
            pygame.draw.rect(self.screen, (180, 180, 200),
                           (zone['x'], zone['y'], zone['width'], zone['height']), 1)

        # Draw rooms with enhanced visibility
        room_colors = {
            RoomType.EMERGENCY: (231, 76, 60),
            RoomType.ICU: (142, 68, 173),
            RoomType.SURGERY: (39, 174, 96),
            RoomType.GENERAL: (52, 152, 219),
            RoomType.LAB: (243, 156, 18),
            RoomType.PHARMACY: (22, 160, 133),
            RoomType.RADIOLOGY: (155, 89, 182),
            RoomType.CARDIOLOGY: (230, 126, 34),
            RoomType.PEDIATRICS: (241, 196, 15),
            RoomType.NEUROLOGY: (52, 73, 94)
        }

        for room in self.room_boundaries:
            color = room_colors.get(room['type'], (100, 100, 100))

            # Draw room with gradient effect
            pygame.draw.rect(self.screen, color,
                           (room['x'], room['y'], room['width'], room['height']))

            # Draw room border
            border_color = (255, 255, 255) if self.carrying_patient and room['type'] == self.carrying_patient['target_room'] else (200, 200, 200)
            border_width = 3 if self.carrying_patient and room['type'] == self.carrying_patient['target_room'] else 2
            pygame.draw.rect(self.screen, border_color,
                           (room['x'], room['y'], room['width'], room['height']), border_width)

            # Draw room label
            room_name = room.get('name', room['type'].name)
            text_surface = self.font_small.render(room_name, True, (255, 255, 255))
            text_rect = text_surface.get_rect()

            text_x = room['x'] + (room['width'] - text_rect.width) // 2
            text_y = room['y'] + (room['height'] - text_rect.height) // 2

            # Enhanced text background
            bg_rect = pygame.Rect(text_x - 3, text_y - 2, text_rect.width + 6, text_rect.height + 4)
            pygame.draw.rect(self.screen, (0, 0, 0), bg_rect)
            pygame.draw.rect(self.screen, border_color, bg_rect, 1)

            self.screen.blit(text_surface, (text_x, text_y))

        # Draw drug stations with enhanced visibility
        for i, (drug_x, drug_y) in enumerate(self.drug_stations):
            if self.drug_stations_available[i]:
                # Check if drug station is in valid corridor
                if not self._is_on_corridor(drug_x, drug_y):
                    # Move drug station to nearest valid corridor position
                    drug_x, drug_y = self._get_nearest_corridor_position(drug_x, drug_y)
                    self.drug_stations[i] = (drug_x, drug_y)  # Update position

                # Pulsing effect for available drug stations
                pulse = int(abs(math.sin(pygame.time.get_ticks() * 0.005)) * 255)
                drug_color = (22, 160, 133) if not self.has_drugs else (100, 100, 100)

                pygame.draw.circle(self.screen, drug_color, (drug_x, drug_y), 12)
                pygame.draw.circle(self.screen, (255, 255, 255), (drug_x, drug_y), 12, 2)

                # Label
                text_surface = self.font_small.render("DRUG", True, (255, 255, 255))
                self.screen.blit(text_surface, (drug_x-15, drug_y-30))

        # Draw patients with enhanced indicators
        patient_colors = {
            PatientUrgency.CRITICAL: (231, 76, 60),
            PatientUrgency.MODERATE: (243, 156, 18),
            PatientUrgency.LOW: (46, 204, 113)
        }

        for patient in self.patients:
            x, y = patient['position']
            color = patient_colors[patient['urgency']]

            # Draw patient with pulsing effect for critical patients
            if patient['urgency'] == PatientUrgency.CRITICAL:
                pulse_size = 12 + int(abs(math.sin(pygame.time.get_ticks() * 0.01)) * 3)
                pygame.draw.circle(self.screen, color, (x, y), pulse_size)
            else:
                pygame.draw.circle(self.screen, color, (x, y), 12)

            # Drug indicator
            if patient['needs_drugs'] and not patient['treated_with_drugs']:
                pygame.draw.circle(self.screen, (255, 255, 255), (x, y), 15, 3)
                # Draw plus sign for drug indicator
                pygame.draw.line(self.screen, (255, 255, 255), (x-8, y), (x+8, y), 2)
                pygame.draw.line(self.screen, (255, 255, 255), (x, y-8), (x, y+8), 2)

            # Draw patient urgency indicator
            urgency_text = patient['urgency'].name[0]
            text_surface = self.font_small.render(urgency_text, True, (255, 255, 255))
            text_rect = text_surface.get_rect(center=(x, y))
            self.screen.blit(text_surface, text_rect)

        # Draw agent with enhanced status indicators
        agent_color = (0, 255, 255)
        pygame.draw.circle(self.screen, agent_color, self.agent_pos, 15)
        pygame.draw.circle(self.screen, (255, 255, 255), self.agent_pos, 15, 3)

        # Status indicators around agent
        if self.carrying_patient:
            # Patient indicator
            pygame.draw.circle(self.screen, (255, 255, 0),
                             (self.agent_pos[0] - 20, self.agent_pos[1] - 20), 8)
            # Draw urgency color
            patient_color = patient_colors[self.carrying_patient['urgency']]
            pygame.draw.circle(self.screen, patient_color,
                             (self.agent_pos[0] - 20, self.agent_pos[1] - 20), 6)

        if self.has_drugs:
            # Drug indicator
            pygame.draw.circle(self.screen, (22, 160, 133),
                             (self.agent_pos[0] + 20, self.agent_pos[1] - 20), 8)
            pygame.draw.circle(self.screen, (255, 255, 255),
                             (self.agent_pos[0] + 20, self.agent_pos[1] - 20), 8, 2)

        # Draw connection lines for better visualization
        if self.carrying_patient:
            # Line to target room
            target_room_type = self.carrying_patient['target_room']
            for room in self.room_boundaries:
                if room['type'] == target_room_type:
                    room_center_x = room['x'] + room['width'] // 2
                    room_center_y = room['y'] + room['height'] // 2
                    pygame.draw.line(self.screen, (255, 255, 0, 100),
                                   self.agent_pos, (room_center_x, room_center_y), 2)
                    break
        elif not self.has_drugs and any(p['needs_drugs'] for p in self.patients):
            # Line to nearest drug station
            nearest_drug = None
            min_dist = float('inf')
            for i, (drug_x, drug_y) in enumerate(self.drug_stations):
                if self.drug_stations_available[i]:
                    dist = math.sqrt((self.agent_pos[0] - drug_x)**2 + (self.agent_pos[1] - drug_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_drug = (drug_x, drug_y)

            if nearest_drug:
                pygame.draw.line(self.screen, (100, 255, 100),
                               self.agent_pos, nearest_drug, 2)
        elif self.patients:
            # Line to nearest patient
            nearest_patient = min(self.patients,
                                key=lambda p: math.sqrt((self.agent_pos[0] - p['position'][0])**2 +
                                                      (self.agent_pos[1] - p['position'][1])**2))
            pygame.draw.line(self.screen, (100, 255, 100),
                           self.agent_pos, nearest_patient['position'], 2)

        # Draw enhanced HUD
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])

    def _render_rgb_array(self):
        """Return RGB array representation of the current state"""
        if not hasattr(self, 'screen'):
            # Create a temporary surface for RGB array rendering
            temp_screen = pygame.Surface((self.WORLD_WIDTH, self.WORLD_HEIGHT))
            # Simplified rendering for RGB array
            temp_screen.fill((30, 30, 50))

            # Draw basic elements
            for room in self.room_boundaries:
                pygame.draw.rect(temp_screen, (100, 100, 100),
                               (room['x'], room['y'], room['width'], room['height']))

            # Draw patients
            for patient in self.patients:
                x, y = patient['position']
                color = (255, 0, 0) if patient['urgency'] == PatientUrgency.CRITICAL else (255, 255, 0)
                pygame.draw.circle(temp_screen, color, (x, y), 10)

            # Draw agent
            pygame.draw.circle(temp_screen, (0, 255, 255), self.agent_pos, 12)

            # Convert to RGB array
            return pygame.surfarray.array3d(temp_screen).swapaxes(0, 1)
        else:
            return pygame.surfarray.array3d(self.screen).swapaxes(0, 1)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()

    def get_action_meanings(self):
        """Return the meaning of each action"""
        return ['UP', 'DOWN', 'LEFT', 'RIGHT', 'UP-LEFT', 'UP-RIGHT', 'DOWN-LEFT', 'DOWN-RIGHT', 'STAY']

# Test and demo functions
def test_environment():
    """Test the environment with random actions"""
    env = HospitalNavigationEnv(render_mode='human', simple_mode=True)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Testing Hospital Navigation Environment")
    print("Press ESC to quit, SPACE to reset")

    while not done and steps < 2000:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    done = True
                elif event.key == pygame.K_SPACE:
                    obs, _ = env.reset()
                    total_reward = 0
                    steps = 0
                    continue

        # Take random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        env.render()

        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}: Reward = {total_reward:.2f}, Patients saved = {env.patients_saved}")

    print(f"Episode finished! Total reward: {total_reward:.2f}, Steps: {steps}, Patients saved: {env.patients_saved}")
    env.close()

def keyboard_control_demo():
    """Demo with keyboard controls"""
    env = HospitalNavigationEnv(render_mode='human', simple_mode=False)
    obs, _ = env.reset()
    done = False
    total_reward = 0

    print("Keyboard Control Demo:")
    print("Arrow keys to move, WASD for diagonal movement")
    print("SPACE to stay still, R to reset, ESC to quit")
    print("\nControls:")
    print("↑ = Up, ↓ = Down, ← = Left, → = Right")
    print("Q = Up-Left, E = Up-Right, Z = Down-Left, C = Down-Right")
    print("SPACE = Stay still")
    print(f"\nDrug stations are at: {env.drug_stations}")
    print("Get close to a drug station to automatically collect drugs!")

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                action = None
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_q:  # Up-Left
                    action = 4
                elif event.key == pygame.K_e:  # Up-Right
                    action = 5
                elif event.key == pygame.K_z:  # Down-Left
                    action = 6
                elif event.key == pygame.K_c:  # Down-Right
                    action = 7
                elif event.key == pygame.K_SPACE:
                    action = 8
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                    total_reward = 0
                    print(f"Environment reset! Drug stations at: {env.drug_stations}")
                    continue
                elif event.key == pygame.K_ESCAPE:
                    done = True

                if action is not None:
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward

                    # Debug information
                    agent_pos = env.agent_pos
                    print(f"Action: {env.get_action_meanings()[action]}, Pos: {agent_pos}, Reward: {reward:.2f}, Total: {total_reward:.2f}")

                    # Check distance to drug stations
                    if not env.has_drugs:
                        for i, (drug_x, drug_y) in enumerate(env.drug_stations):
                            if env.drug_stations_available[i]:
                                dist = math.sqrt((agent_pos[0] - drug_x)**2 + (agent_pos[1] - drug_y)**2)
                                print(f"Distance to drug station {i}: {dist:.1f}")

                    # Status updates
                    if env.has_drugs:
                        print("✓ You have drugs!")
                    if env.carrying_patient:
                        print(f"✓ Carrying patient: {env.carrying_patient['urgency'].name}")

        env.render()

    env.close()


if __name__ == "__main__":
    # Run test
    test_environment()

    # Uncomment to run keyboard demo instead
    keyboard_control_demo()
