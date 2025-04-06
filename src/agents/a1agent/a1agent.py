# Ammaar Melethil | 30141956
# 2023-02-05
# CPSC 383 W25 | T05 | Assignment 1

from typing import override
import heapq
import math

from aegis import (
    END_TURN,
    MOVE,
    SAVE_SURV,
    AgentCommand,
    Direction,
    Survivor,
    create_location,
)
from a1.agent import BaseAgent, Brain, AgentController

def heuristic(a, b) -> float:
    """Euclidean distance heuristic."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

class ExampleAgent(Brain):
    def __init__(self) -> None:
        super().__init__()
        self._agent: AgentController = BaseAgent.get_agent()
        self._path = []
        self._previous_energy = None
        self._previous_location = None
        self._move_costs = {}  # For version 3

    def send_and_end_turn(self, command: AgentCommand):
        """Send a command and end your turn."""
        self._agent.log(f"SENDING {command}")
        self._agent.send(command)
        self._agent.send(END_TURN())

    @override
    def think(self) -> None:
        world = self.get_world()
        current_loc = self._agent.get_location()
        current_energy = self._agent.get_energy_level()

        # Basic checks
        if world is None or self._agent.get_round_number() == 1:
            self.send_and_end_turn(MOVE(Direction.CENTER))
            return

        current_cell = world.get_cell_at(current_loc)
        if current_cell is None:
            self.send_and_end_turn(MOVE(Direction.CENTER))
            return

        # Save survivor if found
        if isinstance(current_cell.get_top_layer(), Survivor):
            self.send_and_end_turn(SAVE_SURV())
            return

        # Detect which version we're running
        is_version_3 = not hasattr(current_cell, 'move_cost')

        # For version 3, track move costs
        if is_version_3 and self._previous_location and self._previous_location != current_loc:
            cost = self._previous_energy - current_energy
            self._move_costs[(self._previous_location.x, self._previous_location.y,
                            current_loc.x, current_loc.y)] = cost
            if cost > 100:  # High cost discovered
                self._path = []  # Force recalculation

        self._previous_energy = current_energy
        self._previous_location = current_loc

        # Find path if needed
        if not self._path:
            # Find survivor
            goal = None
            for x in range(5):
                for y in range(5):
                    cell = world.get_cell_at(create_location(x, y))
                    if cell and cell.has_survivors:
                        goal = cell.location
                        break
                if goal:
                    break

            if goal:
                # A* implementation
                frontier = []
                heapq.heappush(frontier, (0, current_loc))
                came_from = {current_loc: None}
                cost_so_far = {current_loc: 0}

                while frontier:
                    current_cost, current = heapq.heappop(frontier)
                    
                    if current == goal:
                        break

                    for direction in Direction:
                        if direction == Direction.CENTER:
                            continue
                            
                        next_loc = current.add(direction)
                        next_cell = world.get_cell_at(next_loc)
                        
                        if next_cell is None or next_cell.is_fire_cell() or next_cell.is_killer_cell():
                            continue

                        # Calculate move cost based on version
                        if not is_version_3:
                            # Version 1/2: Use actual move cost
                            move_cost = next_cell.move_cost if hasattr(next_cell, 'move_cost') else 1
                        else:
                            # Version 3: Use discovered cost or estimate
                            key = (current.x, current.y, next_loc.x, next_loc.y)
                            move_cost = self._move_costs.get(key, 5)  # Conservative default

                        new_cost = cost_so_far[current] + move_cost
                        if next_loc not in cost_so_far or new_cost < cost_so_far[next_loc]:
                            cost_so_far[next_loc] = new_cost
                            priority = new_cost + heuristic(goal, next_loc)
                            heapq.heappush(frontier, (priority, next_loc))
                            came_from[next_loc] = current

                # Reconstruct path
                self._path = []
                current = goal
                while current is not None:
                    self._path.append(current)
                    current = came_from.get(current)
                self._path.reverse()
                
                # Remove current location
                if self._path and self._path[0] == current_loc:
                    self._path.pop(0)

        # Execute move if we have a path
        if self._path:
            next_loc = self._path[0]
            next_cell = world.get_cell_at(next_loc)
            
            if next_cell and not next_cell.is_fire_cell() and not next_cell.is_killer_cell():
                # Energy safety check
                min_energy = 50 if not is_version_3 else 200
                if current_energy > min_energy:
                    dx = next_loc.x - current_loc.x
                    dy = next_loc.y - current_loc.y
                    
                    for direction in Direction:
                        if direction.dx == dx and direction.dy == dy:
                            self._path.pop(0)
                            self.send_and_end_turn(MOVE(direction))
                            return

        # Default to safe move
        self.send_and_end_turn(MOVE(Direction.CENTER))