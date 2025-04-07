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
        
        elif msg_type == MSG_SAVED:
            # A survivor was saved by another agent
            x, y = int(parts[1]), int(parts[2])
            pos = (x, y)
            if pos in self._known_survivors:
                self._known_survivors.remove(pos)

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
            
            # Check for life signals under rubble
            if ovr.life_signals and ovr.life_signals.size() > 1:  # More than 1 means deeper layers exist
                for i in range(1, ovr.life_signals.size()):  # Start from index 1 (layer under top)
                    signal = ovr.life_signals.get(i)
                    if signal > 0:  # Any positive signal indicates survivor
                        self._agent.log(f"CRITICAL: Survivor under rubble at {pos} - signal:{signal}")
                        self._potential_survivor_rubble.add(pos)
                        # Broadcast to all agents that this rubble has potential survivors
                        self._broadcast(f"{MSG_RUBBLE}:{pos[0]}:{pos[1]}:{energy}:{agents}:1")
                        break

        # Add this line to track that we've moved through a high cost cell
        if ovr.cell_info.move_cost > 2:
            self._last_high_cost_cell = True
        else:
            self._last_high_cost_cell = False

        # CRITICAL FIX 4: Check if we're at a position that previously had rubble but now doesn't
        if pos in self._known_rubble and not isinstance(top_layer, Rubble):
            # The rubble is gone, clear our tracking for this position
            self._agent.log(f"Detected cleared rubble at {pos}")
            self._known_rubble.pop(pos)
            if pos in self._potential_survivor_rubble:
                self._potential_survivor_rubble.remove(pos)
            if pos in self._rubble_leaders:
                del self._rubble_leaders[pos]
            if pos in self._rubble_helpers:
                del self._rubble_helpers[pos]
            
            # Also reset our role if we were working on this rubble
            if self._my_rubble_pos == pos:
                self._my_rubble_role = None
                self._my_rubble_pos = None
                self._rubble_wait_turns = 0

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

    def _handle_rubble_coordination(self, current_pos, current_cell, top_layer, current_energy, current_loc):
        """Simplified rubble handling focused on survivor detection"""
        world = self.get_world()
        if not world:
            return False
            
        # CASE 1: Standing on rubble - check for survivors and dig if needed
        if isinstance(top_layer, Rubble):
            energy_needed = top_layer.remove_energy
            agents_needed = top_layer.remove_agents
            
            # Update tracking
            self._known_rubble[current_pos] = (energy_needed, agents_needed)
            
            # Log rubble information
            self._agent.log(f"On rubble: energy={energy_needed}, agents={agents_needed}, survivor={current_pos in self._potential_survivor_rubble}")
            
            # Always broadcast rubble info, with survivor status if known
            has_survivor = 1 if current_pos in self._potential_survivor_rubble else 0
            self._broadcast(f"{MSG_RUBBLE}:{current_pos[0]}:{current_pos[1]}:{energy_needed}:{agents_needed}:{has_survivor}")

            # CASE A: Zero-agent rubble - always dig
            if agents_needed == 0:
                self._agent.log(f"Digging zero-agent rubble")
                self.send_end_turn(TEAM_DIG())
                return True
                
            # CASE B: Single-agent rubble - dig if enough energy
            if agents_needed == 1:
                # Determine appropriate energy threshold based on world conditions
                # If no charging cells exist, use much lower threshold
                energy_threshold = 10 if not self._known_charging else self._critical_energy
                
                # If rubble contains a survivor, use even lower threshold
                if current_pos in self._potential_survivor_rubble:
                    energy_threshold = 5  # Minimum threshold for critical rescue
                
                # Check energy requirements with adaptive threshold
                if energy_needed == 0 or current_energy > energy_needed + energy_threshold:
                    self._agent.log(f"Digging single-agent rubble with {current_energy} energy (threshold: {energy_threshold})")
                    self.send_end_turn(TEAM_DIG())
                    return True
                else:
                    # Not enough energy, check if we can get help
                    self._broadcast(f"{MSG_RUBBLE_HELPER}:{current_pos[0]}:{current_pos[1]}")
                    self._agent.log(f"Requesting help at {current_pos} - low energy ({current_energy})")
                    
                    # Stay put and wait for help
                    self.send_end_turn(MOVE(Direction.CENTER))
                    return True
            
            # CASE C: Multi-agent rubble - coordinate with other agents
            if agents_needed > 1:
                # Check if enough agents are present on this cell
                agents_here = 1  # Count ourselves
                for _, pos in self._agent_positions.items():
                    if pos == current_pos:
                        agents_here += 1
                
                # Broadcast that we're at this rubble to coordinate 
                self._broadcast(f"{MSG_RUBBLE_HELPER}:{current_pos[0]}:{current_pos[1]}")
                
                # If we have enough agents, dig!
                if agents_here >= agents_needed and (energy_needed == 0 or current_energy > energy_needed + self._critical_energy):
                    self._agent.log(f"DIGGING multi-agent rubble with {agents_here-1} other agents")
                    self.send_end_turn(TEAM_DIG())
                    return True
                
                # Not enough agents yet, wait
                self._agent.log(f"Waiting for more agents at rubble: have {agents_here}, need {agents_needed}")
                self.send_end_turn(MOVE(Direction.CENTER))
                return True
        
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
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return True
                        
        # CASE 4: Move to any unexplored rubble
        for pos, (energy, agents) in self._known_rubble.items():
            if pos not in self._visited:
                path = self._find_path(current_loc, create_location(pos[0], pos[1]))
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self.send_end_turn(MOVE(direction))
                        return True
        
        # No rubble action taken
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
                
                # Next priority: Unchecked rubble (might have survivors)
                elif next_pos not in self._visited:
                    score = 70  # Increased from 50 to prioritize unchecked rubble
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

        # PRIORITY 3: Handle rubble coordination (checking, digging, saving survivors)
        if self._handle_rubble_coordination(current_pos, current_cell, top_layer, current_energy, current_loc):
            return
        
        # PRIORITY 4: Emergency charging when critically low on energy
        if current_energy < self._critical_energy and self._known_charging:
            nearest_charging = self._find_nearest(current_loc, self._known_charging)
            if nearest_charging:
                path = self._find_path(current_loc, nearest_charging)
                if path and len(path) < current_energy - 20:
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
        
        # NEW PRIORITY 9: Actively move to assigned sector for exploration
        if self._my_sector is not None:
            unexplored_loc = self._find_unexplored_in_sector()
            if unexplored_loc:
                path = self._find_path(current_loc, unexplored_loc)
                if path and len(path) + self._critical_energy < current_energy:
                    direction = self._get_next_move_direction(current_loc, path)
                    if direction:
                        self._agent.log(f"Moving to explore sector {self._my_sector}")
                        self.send_end_turn(MOVE(direction))
                        return
        
        # PRIORITY 10: Continue following existing path (was PRIORITY 9 before)
        if self._current_path:
            direction = self._get_next_move_direction(current_loc, self._current_path)
            if direction:
                self.send_end_turn(MOVE(direction))
                return
        
        # PRIORITY 11: General exploration (was PRIORITY 10 before)
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
                
                # Reset rubble commitment if stuck
                self._my_rubble_role = None
                self._my_rubble_pos = None
                
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
                # Factor in agent's ID to encourage different sector choices
                my_id = self._agent.get_agent_id().id 
                # Calculate a bias score based on sector position and agent ID
                # This creates natural distribution based on agent IDs
                x_center = (sector['min_x'] + sector['max_x']) / 2
                y_center = (sector['min_y'] + sector['max_y']) / 2
                id_bias = abs((x_center * my_id + y_center) % self._world_width)
                
                # Add exploration bias to score
                score = sector['explored'] + id_bias * 0.5
                available_sectors.append((i, sector, score))
        
        # Choose sector based on combined score of exploration and agent-specific bias
        if available_sectors:
            available_sectors.sort(key=lambda x: x[2])  # Sort by score
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

