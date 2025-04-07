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
MSG_RUBBLE_LEADER = "LEAD"        # Claim leadership of a rubble site
MSG_RUBBLE_HELPER = "HELP"        # Commit to helping at a rubble site

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
        
        # Add this line here
        self._last_high_cost_cell = False     # Track if we just moved through high-cost terrain
        
        # Energy management
        self._max_energy = 500               # Default max energy
        self._low_energy = 150               # When to seek charging (30%)
        self._critical_energy = 80           # Emergency threshold (16%)
        
        # Path finding
        self._current_path = []              # Path currently following
        self._blacklist = set()              # Temporarily blocked cells
        
        # Agent coordination
        self._agent_positions = {}           # agent_id -> (x,y)
        self._last_position = None           # Last position (for stagnation)
        self._stagnation_counter = 0         # Turns in same position
        
        # Sector exploration
        self._world_width = 15               # Default world size
        self._world_height = 15              # Default world size
        self._sector_size = 4                # Size of exploration sectors
        self._sectors = []                   # List of sector data
        self._my_sector = None               # Currently assigned sector
        self._sectors_explored = set()       # Fully explored sectors
        
        # Rubble coordination
        self._rubble_leaders = {}            # (x,y) -> agent_id
        self._rubble_helpers = defaultdict(set) # (x,y) -> set of agent_ids
        self._my_rubble_role = None          # "leader" or "helper" 
        self._my_rubble_pos = None           # (x,y) of rubble I'm committed to
        self._rubble_wait_turns = 0          # Turns waited at rubble
    
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
                
        elif msg_type == MSG_RUBBLE_LEADER:
            # Another agent is claiming leadership of a rubble cell
            x, y = int(parts[1]), int(parts[2])
            agents_needed = int(parts[3])
            self._rubble_leaders[(x, y)] = sender_id
            
        elif msg_type == MSG_RUBBLE_HELPER:
            # Another agent is helping at a rubble cell
            x, y = int(parts[1]), int(parts[2])
            self._rubble_helpers[(x, y)].add(sender_id)

    @override
    def handle_observe_result(self, ovr: OBSERVE_RESULT) -> None:
        """Process observation results"""
        loc = ovr.cell_info.location
        pos = (loc.x, loc.y)
        
        # Update world size estimates
        self._world_width = max(self._world_width, loc.x + 1)
        self._world_height = max(self._world_height, loc.y + 1)
        
        # Mark position as visited
        self._visited.add(pos)
        
        # Check for dangerous cells in surroundings
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
        
        # Check for rubble
        if isinstance(top_layer, Rubble):
            energy = top_layer.remove_energy
            agents = top_layer.remove_agents
            self._known_rubble[pos] = (energy, agents)
            
            # CRITICAL FIX: Add this position to potential_survivor_rubble IMMEDIATELY if there are ANY signals
            # The first signal is the top layer, we need to check DEEPER layers for survivors under rubble
            if ovr.life_signals and ovr.life_signals.size() > 1:  # More than 1 means deeper layers exist
                for i in range(1, ovr.life_signals.size()):  # Start from index 1 (layer under top)
                    signal = ovr.life_signals.get(i)
                    if signal > 0:  # Any positive signal indicates survivor
                        self._agent.log(f"CRITICAL: Survivor under rubble at {pos} - signal:{signal}")
                        self._potential_survivor_rubble.add(pos)
                        self._broadcast(f"{MSG_RUBBLE}:{pos[0]}:{pos[1]}:{energy}:{agents}:1")
                        break
        
        # More aggressive direct detection
        elif ovr.life_signals and ovr.life_signals.size() > 0:
            for i in range(ovr.life_signals.size()):
                if ovr.life_signals.get(i) > 0:
                    self._agent.log(f"DETECTED DIRECT SURVIVOR at {pos}")
                    self._known_survivors.add(pos)
                    self._broadcast(f"{MSG_SURVIVOR}:{pos[0]}:{pos[1]}")
                    break
        
        # Add this line to track that we've moved through a high cost cell
        if ovr.cell_info.move_cost > 2:
            self._last_high_cost_cell = True
        else:
            self._last_high_cost_cell = False

    def _process_dangerous_cells(self, surround_info):
        """Process and share dangerous cells"""
        for direction in Direction:
            cell_info = surround_info.get_surround_info(direction)
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
        loc = (ssr.surround_info.get_current_info().location.x, 
               ssr.surround_info.get_current_info().location.y)
        
        # Remove from known survivors
        if loc in self._known_survivors:
            self._known_survivors.remove(loc)
        
        # Reset path
        self._current_path = []
        
        # Handle prediction if needed
        if ssr.has_pred_info():
            self._agent.add_prediction_info((ssr.surv_saved_id, ssr.image_to_predict, ssr.all_unique_labels))

    @override
    def handle_predict_result(self, prd: PREDICT_RESULT) -> None:
        pass  # Nothing to do with prediction results

    @override
    def think(self) -> None:
        """Main decision-making method"""
        current_loc = self._agent.get_location()
        current_pos = (current_loc.x, current_loc.y)
        current_energy = self._agent.get_energy_level()
        
        # Share our position
        self._broadcast(f"{MSG_POSITION}:{current_loc.x}:{current_loc.y}")
        
        # Mark position as visited
        self._visited.add(current_pos)
        
        # Check for stagnation
        self._check_stagnation(current_pos)
        
        # Get world information
        world = self.get_world()
        if not world:
            self.send_end_turn(MOVE(Direction.CENTER))
            return
        
        # Manage sector assignments
        self._manage_sectors()
        
        # Handle predictions if needed
        if self._agent.get_prediction_info_size() > 0:
            surv_id, image, labels = self._agent.get_prediction_info()
            if surv_id >= 0 and labels is not None and len(labels) > 0:
                self.send_end_turn(PREDICT(surv_id, labels[0]))
                return
        
        # Get current cell information
        current_cell = world.get_cell_at(current_loc)
        if not current_cell:
            self.send_end_turn(MOVE(Direction.CENTER))
            return
        
        # PRIORITY 1: Save survivor if on one
        top_layer = current_cell.get_top_layer()
        if isinstance(top_layer, Survivor):
            self.send_end_turn(SAVE_SURV())
            return
        
        # PRIORITY 2: Move to adjacent survivor if any
        survivor_direction = self._check_adjacent_cells(current_loc, world, Survivor)
        if survivor_direction:
            self.send_end_turn(MOVE(survivor_direction))
            return

        # PRIORITY 2.5: Check adjacent rubble cells that might contain survivors
        adjacent_rubble = self._check_adjacent_rubble()
        if adjacent_rubble:
            rubble_pos, direction = adjacent_rubble
            self._agent.log(f"Moving to check rubble at {rubble_pos} for survivors")
            self.send_end_turn(MOVE(direction))
            return

        # NEW PRIORITY: Check for unchecked rubble cells nearby
        for pos in self._known_rubble:
            if pos not in self._potential_survivor_rubble and pos not in self._visited:
                # This is rubble we haven't checked for survivors yet
                rubble_loc = create_location(pos[0], pos[1])
                path = self._find_path(current_loc, rubble_loc)
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return

        # NEW PRIORITY: Move to charging cell if adjacent after high move cost cell
        if self._last_high_cost_cell and current_energy < self._max_energy - 100:
            charging_direction = self._check_adjacent_charging_cells(current_loc, world)
            if charging_direction:
                self.send_end_turn(MOVE(charging_direction))
                return
        
        # PRIORITY 3: Handle rubble coordination
        if self._handle_rubble_coordination(current_pos, current_cell, top_layer, current_energy, current_loc):
            return
        
        # PRIORITY 4: Charge if on charging cell and energy low
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
        
        # PRIORITY 5: Emergency charging when critically low on energy
        if current_energy < self._critical_energy and self._known_charging:
            nearest_charging = self._find_nearest(current_loc, self._known_charging)
            if nearest_charging:
                path = self._find_path(current_loc, nearest_charging)
                if path and len(path) < current_energy - 20:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 6: Move to nearest survivor
        if self._known_survivors:
            nearest_survivor = self._find_nearest(current_loc, self._known_survivors)
            if nearest_survivor:
                path = self._find_path(current_loc, nearest_survivor)
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 7: Move to potential survivor rubble
        if self._potential_survivor_rubble:
            for pos in self._potential_survivor_rubble:
                # Check if rubble still needs help
                if pos in self._known_rubble:
                    energy, agents = self._known_rubble[pos]
                    path = self._find_path(current_loc, create_location(pos[0], pos[1]))
                    if path and len(path) + self._critical_energy < current_energy:
                        direction = self._get_next_move_direction(current_loc, path)
                        if direction:
                            self.send_end_turn(MOVE(direction))
                            return
        
        # PRIORITY 8: Seek charging when energy low
        if current_energy < self._low_energy and self._known_charging:
            nearest_charging = self._find_nearest(current_loc, self._known_charging)
            if nearest_charging:
                path = self._find_path(current_loc, nearest_charging)
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 9: Continue following existing path
        if self._current_path:
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

    def _handle_rubble_coordination(self, current_pos, current_cell, top_layer, current_energy, current_loc):
        """Core rubble coordination logic - returns True if action taken"""
        world = self.get_world()
        
        # ---------- CASE 1: We're standing on rubble ----------
        if isinstance(top_layer, Rubble):
            energy_needed = top_layer.remove_energy
            agents_needed = top_layer.remove_agents
            
            self._agent.log(f"On rubble at {current_pos}: energy={energy_needed}, agents={agents_needed}, survivor={current_pos in self._potential_survivor_rubble}")
            
            # CASE 1A: Zero-agent rubble - always dig regardless of energy (it's free)
            if agents_needed == 0:
                self._agent.log(f"Digging zero-agent rubble at {current_pos}")
                self.send_end_turn(TEAM_DIG())
                return True
                
            # CASE 1B: Single-agent rubble
            elif agents_needed == 1:
                # Special case: zero energy required or normal energy check passes
                if energy_needed == 0 or current_energy > energy_needed + self._critical_energy:
                    self._agent.log(f"Digging single-agent rubble at {current_pos}")
                    self.send_end_turn(TEAM_DIG())
                    return True
            
            # CASE 1C: Multi-agent rubble - coordinate using agent IDs
            elif agents_needed > 1:
                my_id = self._agent.get_agent_id().id
                
                # Determine if we should be leader (lowest ID on the cell)
                should_be_leader = True
                for other_id, other_pos in self._agent_positions.items():
                    if other_id < my_id and other_pos == current_pos:
                        should_be_leader = False
                        break
                
                # LEADER ROLE
                if should_be_leader:
                    self._agent.log(f"Becoming leader for rubble at {current_pos}")
                    self._my_rubble_role = "leader"
                    self._my_rubble_pos = current_pos
                    self._rubble_wait_turns = 0
                    self._broadcast(f"{MSG_RUBBLE_LEADER}:{current_pos[0]}:{current_pos[1]}:{agents_needed}")
                    
                    # Critical: Add high urgency flag for survivor rubble
                    urgency = 1 if current_pos in self._potential_survivor_rubble else 0
                    self._broadcast(f"{MSG_RUBBLE}:{current_pos[0]}:{current_pos[1]}:{energy_needed}:{agents_needed}:{urgency}")
                    
                    # Check if we have enough helpers to dig
                    helpers = self._rubble_helpers.get(current_pos, set())
                    helper_count = len(helpers)
                    
                    # CRITICAL FIX: Special case for zero-energy multi-agent rubble
                    if energy_needed == 0:
                        if helper_count + 1 >= agents_needed:  # We have enough helpers
                            self._agent.log(f"DIGGING zero-energy rubble with {helper_count} helpers")
                            self.send_end_turn(TEAM_DIG())
                            return True
                    # Normal energy check for multi-agent rubble
                    elif helper_count + 1 >= agents_needed and current_energy > energy_needed + self._critical_energy:
                        self._agent.log(f"DIGGING rubble with {helper_count} helpers")
                        self.send_end_turn(TEAM_DIG())
                        return True
                    
                    # Stay put as leader
                    self.send_end_turn(MOVE(Direction.CENTER))
                    return True
                    
                # HELPER ROLE
                else:
                    self._agent.log(f"Becoming helper (not lowest ID)")
                    self._my_rubble_role = "helper"
                    self._my_rubble_pos = current_pos
                    self._broadcast(f"{MSG_RUBBLE_HELPER}:{current_pos[0]}:{current_pos[1]}")
                    
                    # Move off to an adjacent cell
                    for direction in sorted(list(Direction), key=lambda _: random.random()):
                        if direction != Direction.CENTER:
                            next_loc = current_loc.add(direction)
                            next_pos = (next_loc.x, next_loc.y)
                            
                            # Check if position is valid and safe
                            if (world.on_map(next_loc) and 
                                next_pos not in self._dangerous_cells and
                                next_pos not in self._blacklist):
                                self.send_end_turn(MOVE(direction))
                                return True
                    
                    # If can't find safe adjacent cell, stay put temporarily
                    self.send_end_turn(MOVE(Direction.CENTER))
                    return True
        
        # ---------- CASE 2: We're committed to helping ----------
        elif self._my_rubble_role == "helper" and self._my_rubble_pos:
            # Check if we're adjacent to our assigned rubble
            rubble_loc = create_location(self._my_rubble_pos[0], self._my_rubble_pos[1])
            dx = abs(current_loc.x - rubble_loc.x)
            dy = abs(current_loc.y - rubble_loc.y)
            
            # If adjacent, stay put and help
            if (dx <= 1 and dy <= 1) and not (dx == 0 and dy == 0):  # Adjacent but not on top
                # Check if the leader still exists
                if self._my_rubble_pos in self._rubble_leaders:
                    self._broadcast(f"{MSG_RUBBLE_HELPER}:{self._my_rubble_pos[0]}:{self._my_rubble_pos[1]}")
                    self.send_end_turn(MOVE(Direction.CENTER))
                    return True
                else:
                    # If no leader, stop being a helper
                    self._my_rubble_role = None
                    self._my_rubble_pos = None
            else:
                # If not adjacent, move adjacent to rubble
                for direction in Direction:
                    if direction == Direction.CENTER:
                        continue
                        
                    next_loc = rubble_loc.add(direction)
                    next_pos = (next_loc.x, next_loc.y)
                    
                    if next_pos == current_pos:  # We're already in a valid position
                        self.send_end_turn(MOVE(Direction.CENTER))
                        return True
                        
                    # Find path to an adjacent position
                    if (world.on_map(next_loc) and 
                        next_pos not in self._dangerous_cells and
                        next_pos != self._my_rubble_pos):  # Don't move onto the rubble
                        path = self._find_path(current_loc, next_loc)
                        if path:
                            direction = self._get_next_move_direction(current_loc, path)
                            if direction:
                                self.send_end_turn(MOVE(direction))
                                return True
        
        # ---------- CASE 3: We're committed to leading ----------
        elif self._my_rubble_role == "leader" and self._my_rubble_pos:
            # If not on our rubble, move to it
            if current_pos != self._my_rubble_pos:
                rubble_loc = create_location(self._my_rubble_pos[0], self._my_rubble_pos[1])
                path = self._find_path(current_loc, rubble_loc)
                if path:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return True
                else:
                    # Can't find path, give up role
                    self._my_rubble_role = None
                    self._my_rubble_pos = None
        
        # No action taken
        return False

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
                
                # Skip if we don't have enough energy
                if self._agent.get_energy_level() <= energy + self._critical_energy:
                    continue
                    
                # Calculate priority score for this rubble
                score = 0
                
                # Highest priority: Known survivor rubble
                if next_pos in self._potential_survivor_rubble:
                    score = 100
                
                # HIGH PRIORITY: Unchecked rubble (might have survivors)
                elif next_pos not in self._visited:
                    score = 50
                    # Add to known rubble so it can be checked
                    self._known_rubble[next_pos] = (energy, agents)
                    self._broadcast(f"{MSG_RUBBLE}:{next_pos[0]}:{next_pos[1]}:{energy}:{agents}:0")
                
                # Medium priority: Single-agent rubble
                elif agents <= 1:
                    score = 20
                
                # Low priority: Multi-agent rubble needing help
                elif agents > 1:
                    helper_count = len(self._rubble_helpers.get(next_pos, set()))
                    if helper_count + 1 < agents:
                        score = 10
                
                # Update best rubble if this is better
                if score > best_score:
                    best_score = score
                    best_rubble = (next_pos, direction)
        
        return best_rubble

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
                
            # Check if there's a survivor on this cell
            top_layer = cell.get_top_layer()
            if isinstance(top_layer, object_type):
                return direction
                
        return None

    def _check_adjacent_charging_cells(self, current_loc, world):
        """Find charging cells adjacent to the current location"""
        for direction in Direction:
            if direction == Direction.CENTER:
                continue
                
            next_loc = current_loc.add(direction)
            if not world.on_map(next_loc):
                continue
                
            cell = world.get_cell_at(next_loc)
            if cell and cell.is_charging_cell():
                return direction
                
        return None

    def _check_stagnation(self, current_pos):
        """Check and handle stagnation"""
        if current_pos == self._last_position:
            self._stagnation_counter += 1
            if self._stagnation_counter > 3:
                self._blacklist.add(current_pos)
                self._current_path = []
                
                # Reset rubble commitment if stuck
                self._my_rubble_role = None
                self._my_rubble_pos = None
                
                self._stagnation_counter = 0
        else:
            self._stagnation_counter = 0
            self._last_position = current_pos

    def _manage_sectors(self):
        """Manage sector assignments"""
        # Create sectors if needed
        if not self._sectors and self._world_width > 0 and self._world_height > 0:
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
        # Find sectors that aren't fully explored
        available_sectors = []
        for i, sector in enumerate(self._sectors):
            if i not in self._sectors_explored and sector['explored'] < 80:
                available_sectors.append((i, sector))
        
        # Choose least explored sector
        if available_sectors:
            available_sectors.sort(key=lambda x: x[1]['explored'])
            self._my_sector = available_sectors[0][0]
        elif self._sectors:
            # If all sectors explored, pick any
            self._my_sector = random.randint(0, len(self._sectors) - 1)

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
        
        # If no more unexplored cells, mark sector as explored
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
        """A* pathfinding with safety checks"""
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
                
                # Calculate move cost
                move_cost = 1
                top_layer = cell.get_top_layer()
                if isinstance(top_layer, Rubble) and (next_loc.x != goal.x or next_loc.y != goal.y):
                    # Higher cost for rubble unless it's our goal
                    move_cost = 3 if top_layer.remove_agents > 1 else 2
                
                # Update path if better
                new_cost = cost_so_far[current] + move_cost
                if next_loc not in cost_so_far or new_cost < cost_so_far[next_loc]:
                    cost_so_far[next_loc] = new_cost
                    priority = new_cost + abs(next_loc.x - goal.x) + abs(next_loc.y - goal.y)
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
    
    def _broadcast(self, message: str) -> None:
        """Send a message to all agents"""
        agent_id_list = AgentIDList()
        self._agent.send(SEND_MESSAGE(agent_id_list, message))

    def send_end_turn(self, command: AgentCommand) -> None:
        """Send a command and end the turn"""
        self._agent.send(command)
        self._agent.send(END_TURN())

