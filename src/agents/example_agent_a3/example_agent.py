# CPSC 383 W25 T05 Assignment 3 | Submission: 2025-04-08
# Ammaar Melethil | 30141956 
# Muhammad Hasham | 30171303 
# Other 2 ppl in our group withdrew from the class :( -- apologies for the bloated code

from typing import override, Dict, List, Set, Tuple, Optional
from collections import deque, defaultdict
import heapq
import random
import math

from aegis import (
    END_TURN,
    SEND_MESSAGE_RESULT,
    MOVE,
    OBSERVE_RESULT,
    PREDICT_RESULT,
    PREDICT,
    SLEEP,
    SAVE_SURV,
    SAVE_SURV_RESULT,
    SEND_MESSAGE,
    TEAM_DIG,
    AgentCommand,
    AgentIDList,
    Direction,
    Rubble,
    Survivor,
    create_location,
)
from a3.agent import BaseAgent, Brain, AgentController

# ---- SIMPLIFIED MESSAGE TYPES ----
MSG_SURVIVOR = "SURVIVOR"         # Share survivor locations
MSG_CHARGING = "CHARGING"         # Share charging cell locations
MSG_RUBBLE = "RUBBLE"             # Share rubble locations
MSG_DANGEROUS = "DANGEROUS"       # Share dangerous cells
MSG_POSITION = "POSITION"         # Share agent positions
MSG_SECTOR = "SECTOR"             # Share which sector an agent is exploring
MSG_RUBBLE_HELPER = "HELP"        # Commit to helping at a rubble site
MSG_SAVED = "SAVED"               # Indicate a survivor was saved

class ExampleAgent(Brain):
    def __init__(self) -> None:
        super().__init__()
        self._agent = BaseAgent.get_agent()
        
        # Core knowledge
        self._known_survivors = set()         # (x,y) locations
        self._known_charging = set()          # (x,y) locations
        self._known_rubble = {}               # (x,y) -> (energy, agents_needed)
        self._potential_survivor_rubble = set() # Rubble cells with life signals
        self._visited = set()                 # (x,y) visited locations
        self._dangerous_cells = set()         # (x,y) dangerous locations
        self._skipped_rubble = set()          # (x,y) rubble we've decided to skip
        
        # Adaptive cost learning
        self._previous_energy = None          # Energy level from previous turn
        self._previous_location = None        # Location from previous turn
        self._move_costs = {}                 # Dictionary to store learned move costs
        self._high_cost_threshold = 10        # Threshold for high movement cost
        
        # Energy management
        self._max_energy = None               # Default max energy
        self._low_energy = None               # When to seek charging (30%)
        self._critical_energy = None          # Emergency threshold (16%)
        
        # Path finding
        self._current_path = []               # Path currently following
        self._blacklist = set()               # Temporarily blocked cells
        
        # Agent coordination
        self._agent_positions = {}            # agent_id -> (x,y)
        self._last_position = None            # Last position (for stagnation)
        self._stagnation_counter = 0          # Turns in same position
        
        # Sector exploration
        self._world_width = None              # Default world size
        self._world_height = None              # Default world size
        self._sector_size = None              # Size of exploration sectors
        self._sectors = []                    # List of sector data
        self._my_sector = None                # Currently assigned sector
        self._sectors_explored = set()        # Fully explored sectors
        
        # Rubble coordination
        self._rubble_helpers = defaultdict(set) # (x,y) -> set of agent_ids
        
        # Corner handling
        self._corner_positions = set()        # Track corner positions
        self._corner_commitment = 0           # Track commitment to corners
        self._max_corner_commitment = 10      # Wait longer at corner rubble
    
    @override
    def handle_send_message_result(self, smr: SEND_MESSAGE_RESULT) -> None:
        """Handle messages from other agents"""
        message = smr.msg
        parts = message.split(":")
        
        if len(parts) < 2:
            return
            
        msg_type = parts[0]
        sender_id = smr.from_agent_id
        
        if msg_type == MSG_SURVIVOR:
            # Add survivor location
            x, y = int(parts[1]), int(parts[2])
            self._known_survivors.add((x, y))
            
        elif msg_type == MSG_CHARGING:
            # Add charging location
            x, y = int(parts[1]), int(parts[2])
            self._known_charging.add((x, y))
            
        elif msg_type == MSG_RUBBLE:
            # Add rubble info and check if it might have survivors
            x, y = int(parts[1]), int(parts[2])
            energy = int(parts[3])
            agents = int(parts[4])
            self._known_rubble[(x, y)] = (energy, agents)
            
            # Check if this rubble has potential survivors
            if len(parts) > 5 and int(parts[5]) == 1:
                self._potential_survivor_rubble.add((x, y))
                # Remove from skipped rubble if it has survivors
                if (x, y) in self._skipped_rubble:
                    self._skipped_rubble.remove((x, y))
            
        elif msg_type == MSG_DANGEROUS:
            # Add dangerous cell
            x, y = int(parts[1]), int(parts[2])
            self._dangerous_cells.add((x, y))
            
        elif msg_type == MSG_POSITION:
            # Track other agent positions
            x, y = int(parts[1]), int(parts[2])
            self._agent_positions[sender_id] = (x, y)
            
        elif msg_type == MSG_SECTOR:
            # Update which sector an agent is exploring
            sector_idx = int(parts[1])
            if sender_id != self._agent.get_agent_id().id:  # Only track other agents
                if sector_idx == self._my_sector:  # If conflict with my sector
                    # The agent with the lower ID keeps the sector
                    if sender_id < self._agent.get_agent_id().id:
                        self._my_sector = None  # I give up this sector
                
        elif msg_type == MSG_RUBBLE_HELPER:
            # Another agent is helping at a rubble cell
            x, y = int(parts[1]), int(parts[2])
            self._rubble_helpers[(x, y)].add(sender_id)
        
        elif msg_type == MSG_SAVED:
            # A survivor was saved by another agent
            x, y = int(parts[1]), int(parts[2])
            pos = (x, y)
            if pos in self._known_survivors:
                self._known_survivors.remove(pos)

    @override
    def handle_observe_result(self, ovr: OBSERVE_RESULT) -> None:
        """Process observation results"""
        loc = ovr.cell_info.location
        pos = (loc.x, loc.y)
        
        # Mark position as visited
        self._visited.add(pos)
        
        # Process dangerous cells
        self._process_dangerous_cells(ovr.surround_info)
        
        # Check for charging cells
        if ovr.cell_info.is_charging_cell():
            self._known_charging.add(pos)
            self._broadcast(f"{MSG_CHARGING}:{pos[0]}:{pos[1]}")
        
        # Process top layer
        top_layer = ovr.cell_info.get_top_layer()
        
        # Check for survivors
        if isinstance(top_layer, Survivor):
            self._known_survivors.add(pos)
            self._broadcast(f"{MSG_SURVIVOR}:{pos[0]}:{pos[1]}")
        
        # Check for rubble and survivors underneath
        if isinstance(top_layer, Rubble):
            energy = top_layer.remove_energy
            agents = top_layer.remove_agents
            self._known_rubble[pos] = (energy, agents)
            
            # Check for survivors under rubble
            has_survivor = self._detect_survivor_under_rubble(ovr, pos)
            
            if has_survivor:
                self._potential_survivor_rubble.add(pos)
                # Remove from skipped list if it was previously added
                if pos in self._skipped_rubble:
                    self._skipped_rubble.remove(pos)
                # Broadcast to all agents that this rubble has potential survivors
                self._broadcast(f"{MSG_RUBBLE}:{pos[0]}:{pos[1]}:{energy}:{agents}:1")

        # Check if we're at a position that previously had rubble but now doesn't
        if pos in self._known_rubble and not isinstance(top_layer, Rubble):
            self._clear_rubble_data(pos)

    def _detect_survivor_under_rubble(self, ovr, pos):
        """Unified method to detect survivors under rubble using multiple methods"""
        # Method 1: Check using life signals
        if ovr.life_signals and ovr.life_signals.size() > 1:
            for i in range(1, ovr.life_signals.size()):
                signal = ovr.life_signals.get(i)
                if signal > 0:
                    self._agent.log(f"CRITICAL: Survivor under rubble at {pos} - via life signal")
                    return True
        
        # Method 2: Use direct cell inspection
        world = self.get_world()
        if world:
            cell = world.get_cell_at(ovr.cell_info.location)
            if cell and cell.number_of_survivors() > 0:
                self._agent.log(f"CRITICAL: Survivor detected under rubble at {pos} - via direct check")
                return True
        
        return False

    def _clear_rubble_data(self, pos):
        """Clear all data related to a rubble position that's now gone"""
        self._agent.log(f"Detected cleared rubble at {pos}")
        self._known_rubble.pop(pos, None)
        self._potential_survivor_rubble.discard(pos)
        self._skipped_rubble.discard(pos)
        self._rubble_helpers.pop(pos, None)

    def _process_dangerous_cells(self, surround_info):
        """Process and share dangerous cells"""
        for direction in Direction:
            if direction == Direction.CENTER:
                continue
                
            cell_info = surround_info.get_info(direction)
            if cell_info:
                loc = cell_info.location
                pos = (loc.x, loc.y)
                
                # Check for dangerous cells
                if cell_info.is_fire_cell() or cell_info.is_killer_cell():
                    self._dangerous_cells.add(pos)
                    self._broadcast(f"{MSG_DANGEROUS}:{pos[0]}:{pos[1]}")

    @override
    def handle_save_surv_result(self, ssr: SAVE_SURV_RESULT) -> None:
        """Handle result of saving a survivor"""
        # Get the location where survivor was saved
        loc = ssr.surround_info.get_current_info().location
        pos = (loc.x, loc.y)
        
        # Log success
        self._agent.log(f"SAVED SURVIVOR at {pos}")
        
        # Remove from known survivors
        if pos in self._known_survivors:
            self._known_survivors.remove(pos)
        
        # Let other agents know this survivor is saved
        self._broadcast(f"{MSG_SAVED}:{pos[0]}:{pos[1]}")
        
        # Handle prediction if needed
        if ssr.has_pred_info():
            self._agent.add_prediction_info((ssr.surv_saved_id, ssr.image_to_predict, ssr.all_unique_labels))
            
    @override
    def handle_predict_result(self, prd: PREDICT_RESULT) -> None:
        """Handle result of prediction"""
        # Log prediction result
        if prd.prediction_correct:
            self._agent.log(f"Correct prediction for survivor {prd.surv_id}")
        else:
            self._agent.log(f"Incorrect prediction for survivor {prd.surv_id}")

    def _check_adjacent_rubble(self):
        """Find important rubble in adjacent cells"""
        current_loc = self._agent.get_location()
        world = self.get_world()
        if not world:
            return None
            
        best_rubble = None
        best_score = -1
        
        # Check adjacent cells for rubble
        for direction in Direction:
            if direction == Direction.CENTER:
                continue
                
            next_loc = current_loc.add(direction)
            next_pos = (next_loc.x, next_loc.y)
            
            if not world.on_map(next_loc) or next_pos in self._dangerous_cells:
                continue
                
            cell = world.get_cell_at(next_loc)
            if not cell:
                continue
            
            top_layer = cell.get_top_layer()
            if isinstance(top_layer, Rubble):
                energy = top_layer.remove_energy
                agents = top_layer.remove_agents
                
                # Check if this is known survivor rubble - NEVER skip these
                has_survivor = next_pos in self._potential_survivor_rubble
                
                # Skip if no survivor AND in our skip list
                if not has_survivor and next_pos in self._skipped_rubble:
                    self._agent.log(f"Skipping previously evaluated rubble at {next_pos}")
                    continue
                
                # Skip if not a survivor AND we shouldn't participate based on ID
                if not has_survivor and agents > 1 and self._agent.get_agent_id().id > agents:
                    self._agent.log(f"Skipping rubble at {next_pos} (need {agents}, ID {self._agent.get_agent_id().id})")
                    # Remember to skip this rubble in the future
                    self._skipped_rubble.add(next_pos)
                    continue
                
                # Calculate priority score for this rubble
                score = 0
                
                # Highest priority: Known survivor rubble
                if has_survivor:
                    score = 100
                    self._agent.log(f"Found SURVIVOR RUBBLE at {next_pos} - agents needed: {agents}")
                # Next priority: Unchecked rubble (might have survivors)
                elif next_pos not in self._visited:
                    score = 90
                    # Add to known rubble so it can be checked
                    self._known_rubble[next_pos] = (energy, agents)
                    self._broadcast(f"{MSG_RUBBLE}:{next_pos[0]}:{next_pos[1]}:{energy}:{agents}:0")
                # Low priority: Already visited rubble
                else:
                    score = 10
                
                # Update best rubble if this is better
                if score > best_score:
                    best_score = score
                    best_rubble = (next_pos, direction)
        
        return best_rubble

    def _handle_rubble_coordination(self, current_pos, current_cell, top_layer, current_energy, current_loc):
        """Handle rubble coordination and digging decisions"""
        world = self.get_world()
        if not world:
            return False
            
        # CASE 1: Standing on rubble - check for survivors and dig if needed
        if isinstance(top_layer, Rubble):
            energy_needed = top_layer.remove_energy
            agents_needed = top_layer.remove_agents
            
            # Update tracking and check for survivors
            self._known_rubble[current_pos] = (energy_needed, agents_needed)
            has_survivor = current_pos in self._potential_survivor_rubble
            
            # Log rubble information
            self._agent.log(f"On rubble: energy={energy_needed}, agents={agents_needed}, survivor={has_survivor}")
            self._broadcast(f"{MSG_RUBBLE}:{current_pos[0]}:{current_pos[1]}:{energy_needed}:{agents_needed}:{1 if has_survivor else 0}")

            # CASE A: Zero-agent rubble - always dig
            if agents_needed == 0:
                self._agent.log(f"Digging zero-agent rubble")
                self.send_end_turn(TEAM_DIG())
                return True
                
            # CASE B: Single-agent rubble - dig if enough energy
            if agents_needed == 1:
                # Determine energy threshold based on conditions
                energy_threshold = 5 if has_survivor else (10 if not self._known_charging else self._critical_energy)
                
                if energy_needed == 0 or current_energy > energy_needed + energy_threshold:
                    self._agent.log(f"Digging single-agent rubble with {current_energy} energy")
                    self.send_end_turn(TEAM_DIG())
                    return True
                else:
                    # Not enough energy, stay put and wait for help
                    self._broadcast(f"{MSG_RUBBLE_HELPER}:{current_pos[0]}:{current_pos[1]}")
                    self.send_end_turn(MOVE(Direction.CENTER))
                    return True
            
            # CASE C: Multi-agent rubble - coordinate with other agents
            if agents_needed > 1:
                # Determine if this agent should participate
                should_participate = self._should_participate_in_rubble(current_pos, agents_needed, has_survivor)
                
                # Count agents present
                agents_here = self._count_agents_at_position(current_pos)
                
                # Special handling for corners
                if self._is_corner_position(current_pos) and energy_needed == 0:
                    if self._handle_corner_rubble(current_pos, agents_needed, agents_here):
                        return True
                
                if should_participate:
                    return self._handle_participate_in_rubble(current_pos, energy_needed, agents_needed, agents_here, current_energy, has_survivor)
                else:
                    # Not participating - add to skip list and move away
                    if not has_survivor:  # Only skip if not a survivor rubble
                        self._skipped_rubble.add(current_pos)
                    
                    # Move to another unexplored area
                    return self._move_to_unexplored_from_rubble(current_loc, world)
        
        # CASE 2-4: Check for other rubble-related actions
        return self._handle_other_rubble_actions(current_loc, current_energy, world)

    def _should_participate_in_rubble(self, pos, agents_needed, has_survivor):
        """Determine if this agent should participate in clearing rubble"""
        # Always participate if rubble has survivors
        if has_survivor:
            return True
        
        # Participate based on agent ID (lowest IDs help)
        return self._agent.get_agent_id().id <= agents_needed

    def _count_agents_at_position(self, pos):
        """Count how many agents are at a specific position"""
        count = 1  # Start with self
        for agent_pos in self._agent_positions.values():
            if agent_pos == pos:
                count += 1
        return count

    def _is_corner_position(self, pos):
        """Check if a position is at a world corner"""
        if self._world_width is None or self._world_height is None:
            return False
            
        return ((pos[0] == 0 and pos[1] == 0) or
                (pos[0] == 0 and pos[1] == self._world_height - 1) or
                (pos[0] == self._world_width - 1 and pos[1] == 0) or
                (pos[0] == self._world_width - 1 and pos[1] == self._world_height - 1))

    def _handle_corner_rubble(self, pos, agents_needed, agents_here):
        """Special handling for corner rubble"""
        self._corner_positions.add(pos)
        self._corner_commitment += 1
        
        # Broadcast stronger signal for corners
        self._broadcast(f"{MSG_RUBBLE_HELPER}:{pos[0]}:{pos[1]}:CORNER")
        
        # Dig if enough agents
        if agents_here >= agents_needed:
            self._agent.log(f"CORNER DIG with {agents_here} agents at {pos}")
            self._corner_commitment = 0  # Reset commitment
            self.send_end_turn(TEAM_DIG())
            return True
        
        # Stay committed to corners longer
        if self._corner_commitment < self._max_corner_commitment:
            self._agent.log(f"CORNER WAIT: {self._corner_commitment}/{self._max_corner_commitment}")
            self.send_end_turn(MOVE(Direction.CENTER))
            return True
        
        # Reset commitment if giving up
        self._corner_commitment = 0
        return False

    def _handle_participate_in_rubble(self, pos, energy_needed, agents_needed, agents_here, current_energy, has_survivor):
        """Handle participation in rubble clearing"""
        # Broadcast that we're helping
        self._broadcast(f"{MSG_RUBBLE_HELPER}:{pos[0]}:{pos[1]}")
        
        # Set energy threshold based on survivor presence
        energy_threshold = 5 if has_survivor else self._critical_energy
        
        # Try to dig if enough agents
        if agents_here >= agents_needed:
            if current_energy > energy_needed + 10:  # Normal dig
                self._agent.log(f"DIGGING multi-agent rubble as agent {self._agent.get_agent_id().id}")
                self.send_end_turn(TEAM_DIG())
                return True
            elif current_energy <= 5:  # Emergency dig
                self._agent.log(f"EMERGENCY DIG with {current_energy} energy")
                self.send_end_turn(TEAM_DIG())
                return True
        
        # Not enough agents yet, wait for help
        self._agent.log(f"Waiting for help at {pos}: have {agents_here}, need {agents_needed}")
        self.send_end_turn(MOVE(Direction.CENTER))
        return True

    def _move_to_unexplored_from_rubble(self, current_loc, world):
        """Move away from rubble to unexplored area"""
        for direction in sorted(list(Direction), key=lambda _: random.random()):
            if direction == Direction.CENTER:
                continue
                
            next_loc = current_loc.add(direction)
            next_pos = (next_loc.x, next_loc.y)
            
            if (world.on_map(next_loc) and 
                next_pos not in self._dangerous_cells and
                next_pos not in self._visited):
                self.send_end_turn(MOVE(direction))
                return True
        
        return False

    def _handle_other_rubble_actions(self, current_loc, current_energy, world):
        """Handle other rubble-related actions (cases 2-4)"""
        # CASE 2: Check adjacent cells for rubble to investigate
        adjacent_rubble = self._check_adjacent_rubble()
        if adjacent_rubble:
            rubble_pos, direction = adjacent_rubble
            self._agent.log(f"Moving to check rubble at {rubble_pos}")
            self.send_end_turn(MOVE(direction))
            return True
            
        # CASE 3: Move to known survivor rubble
        for pos in self._potential_survivor_rubble:
            if pos in self._known_rubble:
                path = self._find_path(current_loc, create_location(pos[0], pos[1]))
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return True
                        
        # CASE 4: Move to any unexplored rubble
        for pos, (energy, agents) in self._known_rubble.items():
            if pos not in self._visited and pos not in self._skipped_rubble:
                path = self._find_path(current_loc, create_location(pos[0], pos[1]))
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return True
        
        # No rubble action taken
        return False

    @override
    def think(self) -> None:
        """Main decision-making method"""
        current_loc = self._agent.get_location()
        current_pos = (current_loc.x, current_loc.y)
        current_energy = self._agent.get_energy_level()
        
        # Learn movement costs - add this new code block
        if self._previous_location and self._previous_energy:
            prev_pos = (self._previous_location.x, self._previous_location.y)
            if prev_pos != current_pos:  # Only if we actually moved
                # Calculate the energy cost of this move
                cost = self._previous_energy - current_energy
                # Store it in our cost dictionary
                self._move_costs[(prev_pos[0], prev_pos[1], current_pos[0], current_pos[1])] = cost
                
                # Log unusually high costs, especially at rubble positions
                if cost > self._high_cost_threshold:
                    self._agent.log(f"HIGH COST MOVE: {prev_pos} -> {current_pos} = {cost}")
                    # Force path recalculation if cost is very high
                    if cost > 20:
                        self._current_path = []
                        
        
        # Store current values for next turn
        self._previous_energy = current_energy
        self._previous_location = current_loc
        
        # Initialize energy thresholds and world data
        self._initialize_settings(current_energy)
        
        # Share position and manage exploration
        self._broadcast(f"{MSG_POSITION}:{current_loc.x}:{current_loc.y}")
        self._visited.add(current_pos)
        self._check_stagnation(current_pos)
        
        # Get world information
        world = self.get_world()
        if not world:
            self.send_end_turn(MOVE(Direction.CENTER))
            return
        
        self._manage_sectors()
        
        # Handle predictions if needed
        if self._handle_predictions():
            return
        
        # Get current cell information
        current_cell = world.get_cell_at(current_loc)
        if not current_cell:
            self.send_end_turn(MOVE(Direction.CENTER))
            return
        
        top_layer = current_cell.get_top_layer()
        
        # PRIORITY 1: Save survivor if on one
        if isinstance(top_layer, Survivor):
            self.send_end_turn(SAVE_SURV())
            return
        
        # PRIORITY 2: Move to adjacent survivor if any
        survivor_direction = self._check_adjacent_cells(current_loc, world, Survivor)
        if survivor_direction:
            self.send_end_turn(MOVE(survivor_direction))
            return
        
        # PRIORITY 2.5: Check adjacent rubble cells
        adjacent_rubble = self._check_adjacent_rubble()
        if adjacent_rubble:
            rubble_pos, direction = adjacent_rubble
            self._agent.log(f"Moving to check rubble at {rubble_pos} for life signals")
            self.send_end_turn(MOVE(direction))
            return

        # PRIORITY 3: Handle rubble coordination
        if self._handle_rubble_coordination(current_pos, current_cell, top_layer, current_energy, current_loc):
            return
        
        # PRIORITY 4-11: Handle other priorities
        self._handle_other_priorities(current_loc, current_pos, current_energy, current_cell, world)

    def _initialize_settings(self, current_energy):
        """Initialize energy thresholds and world settings"""
        # Set energy thresholds if not already set
        if self._max_energy is None:
            self._max_energy = current_energy
            self._low_energy = max(self._max_energy * 0.3, 15)  # 30% or minimum 15
            self._critical_energy = max(self._max_energy * 0.16, 8)  # 16% or minimum 8
        
        # Get world dimensions on first access
        world = self.get_world()
        if world and self._world_width is None:
            self._world_width = world.width
            self._world_height = world.height
            
            # Calculate sector size based on world size
            dimension = min(self._world_width, self._world_height)
            if dimension <= 10:
                self._sector_size = 3
            elif dimension <= 20:
                self._sector_size = 4
            else:
                self._sector_size = 5
            
            # Adjust corner commitment based on world size
            self._max_corner_commitment = max(5, min(15, round(dimension * 0.6)))
            
            self._agent.log(f"World size detected: {self._world_width}x{self._world_height}, "
                           f"using sector size: {self._sector_size}")

    def _handle_predictions(self):
        """Handle any pending predictions"""
        if self._agent.get_prediction_info_size() > 0:
            surv_id, image, labels = self._agent.get_prediction_info()
            if surv_id >= 0 and labels is not None and len(labels) > 0:
                self.send_end_turn(PREDICT(surv_id, labels[0]))
                return True
        return False

    def _handle_other_priorities(self, current_loc, current_pos, current_energy, current_cell, world):
        """Handle priorities 4-11 in think() method"""
        # PRIORITY 4: Emergency charging when critically low on energy
        if current_energy < self._critical_energy and self._known_charging:
            nearest_charging = self._find_nearest(current_loc, self._known_charging)
            if nearest_charging:
                path = self._find_path(current_loc, nearest_charging)
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 5: Charge if on charging cell and energy low
        if current_cell.is_charging_cell():
            self._known_charging.add(current_pos)
            
            # Determine charging threshold
            charge_threshold = self._max_energy - 50  # Default
            
            # If near survivors, use different threshold
            if self._known_survivors:
                nearest = self._find_nearest(current_loc, self._known_survivors)
                if nearest:
                    dist = abs(current_loc.x - nearest.x) + abs(current_loc.y - nearest.y)
                    charge_threshold = min(self._max_energy - 50, self._low_energy + dist * 2)
            
            # Charge if below threshold
            if current_energy < charge_threshold:
                self.send_end_turn(SLEEP())
                return
        
        # PRIORITY 6: Move to nearest survivor
        if self._known_survivors:
            nearest_survivor = self._find_nearest(current_loc, self._known_survivors)
            if nearest_survivor:
                path = self._find_path(current_loc, nearest_survivor)
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 7: Move to potential survivor rubble (already handled in rubble coordination)
        
        # PRIORITY 8: Seek charging when energy low
        if current_energy < self._low_energy and self._known_charging:
            nearest_charging = self._find_nearest(current_loc, self._known_charging)
            if nearest_charging:
                path = self._find_path(current_loc, nearest_charging)
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 9: Actively move to assigned sector for exploration
        if self._my_sector is not None:
            unexplored_loc = self._find_unexplored_in_sector()
            if unexplored_loc:
                path = self._find_path(current_loc, unexplored_loc)
                path_cost = self._calculate_path_energy_cost(path, world)
                if path and path_cost + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self._agent.log(f"Moving to explore sector {self._my_sector}")
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 10: Continue following existing path
        if self._current_path:
            direction = self._get_next_move_direction(current_loc, self._current_path)
            if direction:
                self.send_end_turn(MOVE(direction))
                return
        
        # PRIORITY 11: General exploration
        for direction in sorted(list(Direction), key=lambda _: random.random()):
            if direction == Direction.CENTER:
                continue
                
            next_loc = current_loc.add(direction)
            next_pos = (next_loc.x, next_loc.y)
            
            # Check if valid and unvisited
            if (world.on_map(next_loc) and 
                next_pos not in self._dangerous_cells and 
                next_pos not in self._blacklist):
                
                cell = world.get_cell_at(next_loc)
                if cell and not cell.is_fire_cell() and not cell.is_killer_cell():
                    if next_pos not in self._visited:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # FALLBACK: Move randomly but safely
        safe_directions = []
        for direction in Direction:
            if direction != Direction.CENTER:
                next_loc = current_loc.add(direction)
                next_pos = (next_loc.x, next_loc.y)
                if (world.on_map(next_loc) and 
                    next_pos not in self._dangerous_cells and 
                    next_pos not in self._blacklist):
                    safe_directions.append(direction)
        
        if safe_directions:
            self.send_end_turn(MOVE(random.choice(safe_directions)))
        else:
            # If no safe moves, stay put
            self.send_end_turn(MOVE(Direction.CENTER))

    def _broadcast(self, message: str) -> None:
        """Send a message to all agents"""
        agent_id_list = AgentIDList()
        self._agent.send(SEND_MESSAGE(agent_id_list, message))

    def send_end_turn(self, command: AgentCommand) -> None:
        """Send a command and end the turn"""
        self._agent.send(command)
        self._agent.send(END_TURN())

    def _check_stagnation(self, current_pos):
        """Check and handle stagnation"""
        if current_pos == self._last_position:
            self._stagnation_counter += 1
            if self._stagnation_counter > 3:
                self._blacklist.add(current_pos)
                self._current_path = []
                
                # Reset skipped rubble for this position to reassess
                if current_pos in self._skipped_rubble:
                    self._skipped_rubble.remove(current_pos)
                
                self._stagnation_counter = 0
        else:
            self._stagnation_counter = 0
            self._last_position = current_pos

    def _check_adjacent_cells(self, current_loc, world, object_type):
        """Check surrounding cells for specific object type"""
        for direction in Direction:
            if direction == Direction.CENTER:
                continue
                
            next_loc = current_loc.add(direction)
            if not world.on_map(next_loc):
                continue
                
            cell = world.get_cell_at(next_loc)
            if not cell:
                continue
                
            # Check if there's an object of the specified type on this cell
            top_layer = cell.get_top_layer()
            if isinstance(top_layer, object_type):
                return direction
                
        return None

    def _find_nearest(self, from_loc, locations):
        """Find nearest location from a set of (x,y) tuples"""
        nearest = None
        min_dist = float('inf')
        
        for x, y in locations:
            dist = abs(from_loc.x - x) + abs(from_loc.y - y)
            if dist < min_dist:
                min_dist = dist
                nearest = create_location(x, y)
                
        return nearest

    def _find_path(self, start, goal):
        """A* pathfinding with adaptive move cost learning"""
        world = self.get_world()
        if not world:
            return []
            
        # Initialize search
        frontier = [(0, start)]
        came_from = {start: None}
        cost_so_far = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            
            if current.x == goal.x and current.y == goal.y:
                break
                
            for direction in Direction:
                if direction == Direction.CENTER:
                    continue
                    
                next_loc = current.add(direction)
                next_pos = (next_loc.x, next_loc.y)
                
                # Skip if invalid
                if (not world.on_map(next_loc) or 
                    next_pos in self._blacklist or 
                    next_pos in self._dangerous_cells):
                    continue
                
                # Skip dangerous cells
                cell = world.get_cell_at(next_loc)
                if not cell or cell.is_fire_cell() or cell.is_killer_cell():
                    self._dangerous_cells.add(next_pos)
                    continue
                
                # Calculate move cost - modified to use adaptive learning
                # Check if we've learned the cost for this specific move
                current_pos = (current.x, current.y)
                move_cost_key = (current_pos[0], current_pos[1], next_pos[0], next_pos[1])
                
                # Use learned cost if available, otherwise fallback to cell.move_cost
                if move_cost_key in self._move_costs:
                    move_cost = self._move_costs[move_cost_key]
                    # Apply penalty to cells with previously observed high cost
                    if move_cost > self._high_cost_threshold:
                        move_cost *= 1.5  # Penalize high-cost moves to avoid them
                else:
                    # Default to cell's move cost
                    move_cost = cell.move_cost
                    
                # Update path if better
                new_cost = cost_so_far[current] + move_cost
                if next_loc not in cost_so_far or new_cost < cost_so_far[next_loc]:
                    cost_so_far[next_loc] = new_cost
                    # Use Euclidean distance for heuristic
                    priority = new_cost + math.sqrt((next_loc.x - goal.x)**2 + (next_loc.y - goal.y)**2)
                    heapq.heappush(frontier, (priority, next_loc))
                    came_from[next_loc] = current
        
        # Reconstruct path
        if goal not in came_from:
            return []
            
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        
        # Store the path
        self._current_path = path
        return path

    def _get_next_move_direction(self, current_loc, path):
        """Get direction for next move in path"""
        if not path:
            return None
            
        next_loc = path[0]
        dx = next_loc.x - current_loc.x
        dy = next_loc.y - current_loc.y
        
        for direction in Direction:
            if direction.dx == dx and direction.dy == dy:
                path.pop(0)  # Remove location we're moving to
                return direction
                
        return None

    def _manage_sectors(self):
        """Manage sector assignments"""
        # Create sectors if needed
        if not self._sectors and self._world_width and self._world_height:
            self._create_sectors()
        
        # Choose a sector if needed
        if self._my_sector is None and self._sectors:
            self._choose_sector()
        
        # Share sector info
        if self._my_sector is not None:
            self._broadcast(f"{MSG_SECTOR}:{self._my_sector}")

    def _create_sectors(self):
        """Create exploration sectors"""
        sector_size = self._sector_size
        
        sectors = []
        for x in range(0, self._world_width, sector_size):
            for y in range(0, self._world_height, sector_size):
                sector = {
                    'min_x': x,
                    'max_x': min(x + sector_size - 1, self._world_width - 1),
                    'min_y': y,
                    'max_y': min(y + sector_size - 1, self._world_height - 1),
                    'explored': self._calculate_explored_percentage(x, y, 
                                min(x + sector_size - 1, self._world_width - 1),
                                min(y + sector_size - 1, self._world_height - 1))
                }
                sectors.append(sector)
        
        self._sectors = sectors

    def _choose_sector(self):
        """Choose a sector to explore"""
        # Check if corners need to be explored
        corner_sectors = []
        for i, sector in enumerate(self._sectors):
            # Identify sectors containing corners
            if ((sector['min_x'] == 0 and sector['min_y'] == 0) or
                (sector['min_x'] == 0 and sector['max_y'] == self._world_height - 1) or
                (sector['max_x'] == self._world_width - 1 and sector['min_y'] == 0) or
                (sector['max_x'] == self._world_width - 1 and sector['max_y'] == self._world_height - 1)):
                corner_sectors.append((i, sector, 0))
        
        # Assign to corner first if available
        if corner_sectors:
            self._my_sector = corner_sectors[self._agent.get_agent_id().id % len(corner_sectors)][0]
            return

        # Find sectors that aren't fully explored
        available_sectors = []
        for i, sector in enumerate(self._sectors):
            if i not in self._sectors_explored and sector['explored'] < 80:
                # Calculate bias based on agent ID
                my_id = self._agent.get_agent_id().id 
                x_center = (sector['min_x'] + sector['max_x']) / 2
                y_center = (sector['min_y'] + sector['max_y']) / 2
                id_bias = abs((x_center * my_id + y_center) % self._world_width)
                
                # Calculate score considering exploration status and ID bias
                score = sector['explored'] + id_bias * 0.5
                available_sectors.append((i, sector, score))
        
        if available_sectors:
            available_sectors.sort(key=lambda x: x[2])
            self._my_sector = available_sectors[0][0]
        elif self._sectors:
            # If all sectors explored, pick based on agent ID
            my_id = self._agent.get_agent_id().id
            self._my_sector = my_id % len(self._sectors)

    def _find_unexplored_in_sector(self):
        """Find unexplored location in assigned sector"""
        if self._my_sector is None or not self._sectors:
            return None
            
        sector = self._sectors[self._my_sector]
        min_x, max_x = sector['min_x'], sector['max_x']
        min_y, max_y = sector['min_y'], sector['max_y']
        
        # Find unexplored cells in this sector
        unexplored_cells = []
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) not in self._visited and (x, y) not in self._dangerous_cells:
                    unexplored_cells.append((x, y))
        
        # Mark sector as explored if no unexplored cells
        if not unexplored_cells:
            self._sectors_explored.add(self._my_sector)
            self._my_sector = None
            return None
            
        # Find nearest unexplored cell
        current_loc = self._agent.get_location()
        nearest = None
        min_dist = float('inf')
        
        for x, y in unexplored_cells:
            dist = abs(current_loc.x - x) + abs(current_loc.y - y)
            if dist < min_dist:
                min_dist = dist
                nearest = create_location(x, y)
                
        return nearest

    def _calculate_explored_percentage(self, min_x, min_y, max_x, max_y):
        """Calculate percentage of sector explored"""
        total_cells = (max_x - min_x + 1) * (max_y - min_y + 1)
        explored_cells = 0
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in self._visited:
                    explored_cells += 1
        
        return (explored_cells / total_cells) * 100 if total_cells > 0 else 0

    def _calculate_path_energy_cost(self, path, world):
        """Calculate the energy cost to traverse a path"""
        if not path:
            return 0
            
        total_cost = 0
        for loc in path:
            cell = world.get_cell_at(loc)
            if cell:
                total_cost += cell.move_cost
                
        return total_cost
