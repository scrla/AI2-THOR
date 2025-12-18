from ai2thor.controller import Controller
from collections import defaultdict
import time
import math
import json
from typing import List, Dict, Optional, Tuple, Set
import cv2
import threading
import numpy as np
import heapq

import openai

def call_openai_api(prompt: str, system_prompt: str = "") -> str:
    # api 호출
    try:
        client = openai.OpenAI(api_key="YOUR_API_KEY")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API Error: {str(e)}")
        return ""

# ----------------------------
# A* 알고리즘
# ----------------------------
class AStarPathfinder:
    """A* 알고리즘 기반 경로 탐색"""
    
    def __init__(self, controller, grid_size: float = 0.25):
        self.controller = controller
        self.grid_size = grid_size
        self.reachable_positions_cache = {}
    
    def get_reachable_positions(self, agent_id: int, force_refresh: bool = False) -> Set[Tuple[float, float]]:
        """도달 가능한 모든 그리드 포지션 가져옴"""
        if not force_refresh and agent_id in self.reachable_positions_cache:
            return self.reachable_positions_cache[agent_id]
        
        print(f"[A*] Computing reachable positions for agent {agent_id}...")
        try:
            with controller_lock:
                ev = self.controller.step({"action": "GetReachablePositions", "agentId": agent_id})
            
            reachable = set()
            for pos in ev.metadata["actionReturn"]:
                reachable.add((round(pos["x"] / self.grid_size) * self.grid_size, 
                              round(pos["z"] / self.grid_size) * self.grid_size))
            
            self.reachable_positions_cache[agent_id] = reachable
            print(f"[A*] Found {len(reachable)} reachable positions")
            return reachable
        except Exception as e:
            print(f"[A*] Error getting reachable positions: {str(e)[:80]}")
            return set()
    
    def heuristic(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos: Tuple[float, float], reachable: Set[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """현위치에서 이동 가능한 이웃 노드들"""
        x, z = pos
        neighbors = []
        
        for dx, dz in [(0, self.grid_size), (0, -self.grid_size), 
                       (self.grid_size, 0), (-self.grid_size, 0)]:
            new_x = round((x + dx) / self.grid_size) * self.grid_size
            new_z = round((z + dz) / self.grid_size) * self.grid_size
            new_pos = (new_x, new_z)
            
            if new_pos in reachable:
                neighbors.append(new_pos)
        
        for dx, dz in [(self.grid_size, self.grid_size), (self.grid_size, -self.grid_size),
                       (-self.grid_size, self.grid_size), (-self.grid_size, -self.grid_size)]:
            new_x = round((x + dx) / self.grid_size) * self.grid_size
            new_z = round((z + dz) / self.grid_size) * self.grid_size
            new_pos = (new_x, new_z)
            
            if new_pos in reachable:
                neighbors.append(new_pos)
        
        return neighbors
    
    def find_path(self, agent_id: int, start_pos: Tuple[float, float], 
                  goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        
        start = (round(start_pos[0] / self.grid_size) * self.grid_size,
                round(start_pos[1] / self.grid_size) * self.grid_size)
        goal = (round(goal_pos[0] / self.grid_size) * self.grid_size,
               round(goal_pos[1] / self.grid_size) * self.grid_size)
        
        reachable = self.get_reachable_positions(agent_id)
        
        if not reachable:
            return None
        
        if start not in reachable:
            min_dist = float('inf')
            nearest_start = None
            for pos in reachable:
                dist = self.heuristic(pos, start)
                if dist < min_dist:
                    min_dist = dist
                    nearest_start = pos
            start = nearest_start
        
        if goal not in reachable:
            min_dist = float('inf')
            nearest_goal = None
            for pos in reachable:
                dist = self.heuristic(pos, goal)
                if dist < min_dist:
                    min_dist = dist
                    nearest_goal = pos
            goal = nearest_goal
        
        if start is None or goal is None:
            return None
        
        # A* 알고리즘
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        explored = 0
        while open_set:
            explored += 1
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current, reachable):
                dx = abs(neighbor[0] - current[0])
                dz = abs(neighbor[1] - current[1])
                move_cost = math.sqrt(dx**2 + dz**2)
                
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None

# ----------------------------
# LLM 기반 작업 분해
# ----------------------------
def decompose_task_with_llm(task_text: str, scene_objects: List[str]) -> List[Dict]:
    """LLM을 사용하여 고수준 작업을 subtask로 분해"""
    
    system_prompt = """You are a task planning system for multi-agent robotics in a household environment.
Your job is to decompose high-level tasks into atomic subtasks with proper sequencing.

Available actions:
- GotoObject: Navigate to an object
- PickupObject: Pick up an object (requires the object to be pickupable)
- PutObject: Place held object into/onto target
- OpenObject: Open an openable object (fridge, microwave, cabinet, etc.)
- CloseObject: Close an openable object
- ToggleObjectOn: Turn on a toggleable object
- ToggleObjectOff: Turn off a toggleable object

Important rules:
1. Before picking up an object, the agent must go to it (GotoObject)
2. Before putting an object somewhere, the agent must be holding it
3. To put something in a closed container, it must be opened first
4. Each subtask should have a "chain" identifier to group related sequential tasks
5. Specify required_distance for GotoObject (1.0-1.5 typically)

Return ONLY a valid JSON array of subtasks, no other text."""

    prompt = f"""High-level task: "{task_text}"

Available objects in scene: {', '.join(scene_objects)}

Decompose this task into atomic subtasks. Each subtask should have:
- action: one of the available actions
- object: the object to interact with (if applicable)
- target: the target location for PutObject (if applicable)
- chain: identifier to group sequential tasks
- required_distance: distance needed for GotoObject (default 1.2)

Example format:
[
  {{"action": "GotoObject", "object": "Apple", "chain": "apple_bowl", "required_distance": 1.2}},
  {{"action": "PickupObject", "object": "Apple", "chain": "apple_bowl"}},
  {{"action": "GotoObject", "object": "Bowl", "chain": "apple_bowl", "required_distance": 1.0}},
  {{"action": "PutObject", "target": "Bowl", "chain": "apple_bowl"}}
]

Now decompose the given task:"""

    response = call_openai_api(prompt, system_prompt)
    
    try:
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()
        
        subtasks = json.loads(json_str)
        
        for idx, task in enumerate(subtasks):
            task["task_id"] = idx
        
        print(f"\n[LLM] Decomposed into {len(subtasks)} subtasks:")
        for task in subtasks:
            obj = task.get('object') or task.get('target', '')
            print(f"  Task {task['task_id']}: {task['action']} - {obj} (chain: {task.get('chain', 'none')})")
        
        return subtasks
    
    except json.JSONDecodeError as e:
        print(f"[LLM] Failed to parse JSON response: {e}")
        return []

def analyze_dependencies_with_llm(subtasks: List[Dict]) -> Dict[int, List[int]]:
    """LLM을 사용하여 작업 간 의존성 분석"""
    
    system_prompt = """You are analyzing task dependencies for multi-agent execution.
Tasks in the same chain must be executed sequentially.
Return ONLY a valid JSON object mapping task IDs to their dependencies."""

    task_summary = []
    for task in subtasks:
        obj = task.get('object') or task.get('target', '')
        task_summary.append(f"Task {task['task_id']}: {task['action']} {obj} [chain: {task.get('chain', 'none')}]")
    
    prompt = f"""Analyze dependencies between these tasks:

{chr(10).join(task_summary)}

Rules:
1. Tasks in the same chain must execute in order
2. A task depends on all previous tasks in its chain
3. Return a JSON object where keys are task IDs (as strings) and values are arrays of prerequisite task IDs

Example:
{{
  "1": [0],
  "2": [1],
  "5": [4]
}}

If a task has no dependencies, either omit it or use an empty array.
Return ONLY the JSON object, no other text:"""

    response = call_openai_api(prompt, system_prompt)
    
    try:
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()
        
        deps_dict = json.loads(json_str)
        
        dependencies = defaultdict(list)
        for task_id_str, dep_list in deps_dict.items():
            task_id = int(task_id_str)
            dependencies[task_id] = [int(d) for d in dep_list]
        
        print(f"\n[LLM] Dependencies analyzed:")
        for task_id, deps in dependencies.items():
            if deps:
                print(f"  Task {task_id} depends on: {deps}")
        
        return dependencies
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[LLM] Failed to parse dependencies: {e}")
        return build_dependency_from_chains(subtasks)

def build_dependency_from_chains(subtasks: List[Dict]) -> Dict[int, List[int]]:
    """Chain 기반 의존성 생성"""
    dependencies = defaultdict(list)
    chain_last_task = {}
    
    for task in subtasks:
        task_id = task["task_id"]
        chain = task.get("chain")
        
        if chain:
            if chain in chain_last_task:
                dependencies[task_id].append(chain_last_task[chain])
            chain_last_task[chain] = task_id
    
    return dependencies

def allocate_tasks_with_llm(subtasks: List[Dict], robots: List[Dict]) -> List[Tuple[int, Dict]]:
    """LLM을 사용하여 작업을 로봇에 할당"""
    
    system_prompt = """You are a task allocation system for multi-agent robotics.
Assign tasks to robots based on their capabilities.
Return ONLY a valid JSON array of assignments."""

    # 로봇
    robot_info = []
    for robot in robots:
        skills_str = ", ".join(robot["skills"])
        robot_info.append(f"Robot {robot['id']}: {skills_str}")
    
    # 작업
    task_info = []
    for task in subtasks:
        obj = task.get('object') or task.get('target', '')
        task_info.append(f"Task {task['task_id']}: {task['action']} {obj} [chain: {task.get('chain', 'none')}]")
    
    prompt = f"""Available robots:
{chr(10).join(robot_info)}

Tasks to assign:
{chr(10).join(task_info)}

Rules:
1. Each robot can only perform actions in their skill set
2. Tasks in the same chain should preferably be assigned to the same robot
3. Distribute workload evenly when possible

Return a JSON array of assignments:
[
  {{"task_id": 0, "agent_id": 0}},
  {{"task_id": 1, "agent_id": 0}},
  {{"task_id": 2, "agent_id": 1}}
]

Return ONLY the JSON array, no other text:"""

    response = call_openai_api(prompt, system_prompt)
    
    try:
        json_str = response.strip()
        if json_str.startswith("```"):
            json_str = json_str.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
        json_str = json_str.strip()
        
        assignments = json.loads(json_str)
        
        # 작업 할당 결과
        allocation = []
        task_dict = {task["task_id"]: task for task in subtasks}
        
        for assignment in assignments:
            task_id = assignment["task_id"]
            agent_id = assignment["agent_id"]
            
            if task_id in task_dict:
                task = task_dict[task_id]
                allocation.append((agent_id, task))
        
        print(f"\n[LLM] Task allocation:")
        for agent_id, task in allocation:
            obj = task.get('object') or task.get('target', '')
            print(f"  Task {task['task_id']}: Agent {agent_id} -> {task['action']} {obj}")
        
        return allocation
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[LLM] Failed to parse allocation: {e}")
        # Fallback: 간단한 스킬 기반 작업 할당
        return allocate_tasks_simple(subtasks, robots)

def allocate_tasks_simple(subtasks: List[Dict], robots: List[Dict]) -> List[Tuple[int, Dict]]:
    """간단한 스킬 기반 작업 할당 (fallback)"""
    allocation = []
    chain_agents = {}
    
    for task in subtasks:
        action = task["action"]
        chain = task.get("chain")
        
        if chain and chain in chain_agents:
            agent_id = chain_agents[chain]
        else:
            candidates = [r for r in robots if action in r["skills"]]
            agent_id = candidates[0]["id"] if candidates else 0
            
            if chain:
                chain_agents[chain] = agent_id
        
        allocation.append((agent_id, task))
    
    return allocation

# ----------------------------
# 사용자 입력 함수
# ----------------------------
def get_user_input():
    """사용자로부터 FloorPlan과 작업 입력 받기"""
    print("\n" + "="*60)
    print("AI2-THOR Multi-Agent Task Planner")
    print("="*60)
    
    # FloorPlan 입력 받기
    while True:
        floor_plan = input("\nEnter FloorPlan (e.g., FloorPlan6): ").strip()
        if floor_plan:
            if not floor_plan.endswith("_physics"):
                floor_plan += "_physics"
            break
        print("Please enter a valid FloorPlan name.")
    
    # 에이전트 수 입력 받기
    while True:
        try:
            agent_count = input("\nNumber of agents (1-4, default 3): ").strip()
            if not agent_count:
                agent_count = 3
            else:
                agent_count = int(agent_count)
            
            if 1 <= agent_count <= 4:
                break
            print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")
    
    # High-level task 입력 받기
    while True:
        task = input("\nEnter your high-level task: ").strip()
        if task:
            break
        print("Please enter a task description.")
    
    return floor_plan, agent_count, task

# ----------------------------
# 유틸리티 함수들
# ----------------------------
controller = None
controller_lock = threading.Lock()
pathfinder = None
ROBOTS = []

def get_agent_position(agent_id: int) -> Tuple[float, float, float]:
    ev = controller.last_event
    agent_meta = ev.events[agent_id].metadata
    pos = agent_meta["agent"]["position"]
    return (pos["x"], pos["y"], pos["z"])

def get_agent_rotation(agent_id: int) -> float:
    ev = controller.last_event
    agent_meta = ev.events[agent_id].metadata
    return agent_meta["agent"]["rotation"]["y"]

def calculate_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)

def find_object_in_scene(object_name: str) -> Optional[Dict]:
    ev = controller.last_event
    for o in ev.metadata["objects"]:
        if object_name.lower() == o["objectType"].lower():
            return o
        if object_name.lower() in o["objectType"].lower():
            return o
    return None

def get_visible_objects(agent_id: int) -> List[str]:
    ev = controller.last_event
    agent_event = ev.events[agent_id]
    return [o["objectId"] for o in agent_event.metadata["objects"] if o.get("visible", False)]

def is_object_visible(agent_id: int, object_id: str) -> bool:
    return object_id in get_visible_objects(agent_id)

def look_for_object(agent_id: int, object_id: str, max_rotations: int = 8) -> bool:
    for i in range(max_rotations):
        if is_object_visible(agent_id, object_id):
            return True
        safe_step(agent_id, {"action": "RotateRight", "degrees": 45}, verbose=False)
    return False

def calculate_angle_to_target(agent_pos: Tuple[float, float, float], 
                               agent_rotation: float,
                               target_pos: Tuple[float, float, float]) -> float:
    dx = target_pos[0] - agent_pos[0]
    dz = target_pos[2] - agent_pos[2]
    target_angle = math.degrees(math.atan2(dx, dz))
    relative_angle = (target_angle - agent_rotation + 180) % 360 - 180
    return relative_angle

def rotate_to_face_target(agent_id: int, target_pos: Tuple[float, float, float]) -> bool:
    agent_pos = get_agent_position(agent_id)
    agent_rot = get_agent_rotation(agent_id)
    relative_angle = calculate_angle_to_target(agent_pos, agent_rot, target_pos)
    rotations_needed = round(relative_angle / 45)
    
    for _ in range(abs(rotations_needed)):
        if rotations_needed > 0:
            safe_step(agent_id, {"action": "RotateRight", "degrees": 45}, verbose=False)
        else:
            safe_step(agent_id, {"action": "RotateLeft", "degrees": 45}, verbose=False)
    return True

def navigate_to_position_astar(agent_id: int, target_pos: Tuple[float, float, float], 
                               object_name: str = "target", max_retries: int = 3) -> bool:
    
    for retry in range(max_retries):
        if retry > 0:
            print(f"[Agent {agent_id}] Retry {retry}/{max_retries} for {object_name}")
            pathfinder.get_reachable_positions(agent_id, force_refresh=True)
        
        agent_pos = get_agent_position(agent_id)
        start = (agent_pos[0], agent_pos[2])
        goal = (target_pos[0], target_pos[2])
        
        path = pathfinder.find_path(agent_id, start, goal)
        
        if not path:
            if retry < max_retries - 1:
                time.sleep(0.3)
                continue
            print(f"[Agent {agent_id}] A* failed, using direct navigation")
            return navigate_to_position(agent_id, target_pos, object_name)
        
        print(f"[Agent {agent_id}] Following A* path to {object_name} ({len(path)} waypoints)")
        
        blocked_count = 0
        for i, waypoint in enumerate(path[1:], 1):
            waypoint_3d = (waypoint[0], agent_pos[1], waypoint[1])
            
            rotate_to_face_target(agent_id, waypoint_3d)
            
            success = safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
            
            if not success:
                blocked_count += 1
                if blocked_count > 2:
                    print(f"[Agent {agent_id}] Path blocked multiple times, replanning...")
                    current_pos = get_agent_position(agent_id)
                    remaining_path = pathfinder.find_path(
                        agent_id, 
                        (current_pos[0], current_pos[2]), 
                        goal
                    )
                    if remaining_path and len(remaining_path) > 1:
                        path = remaining_path
                        blocked_count = 0
                    else:
                        break
        
        final_pos = get_agent_position(agent_id)
        final_dist = calculate_distance(final_pos, target_pos)
        
        if final_dist <= 1.5:
            print(f"[Agent {agent_id}] Reached {object_name} ({final_dist:.2f}m)")
            return True
        elif retry < max_retries - 1:
            time.sleep(0.2)
            continue
    
    # fallback
    return navigate_to_position(agent_id, target_pos, object_name)

def navigate_to_position(agent_id: int, target_pos: Tuple[float, float, float], 
                        object_name: str = "target", 
                        required_distance: float = 1.5,
                        max_steps: int = 50) -> bool:
    """직접 네비게이션 (fallback)"""
    print(f"[Agent {agent_id}] Direct navigation to {object_name}...")
    
    stuck_count = 0
    prev_distance = float('inf')
    no_progress_count = 0
    
    for step in range(max_steps):
        agent_pos = get_agent_position(agent_id)
        distance = calculate_distance(agent_pos, target_pos)
        
        if distance <= required_distance:
            print(f"[Agent {agent_id}] Reached {object_name} ({distance:.2f}m)")
            return True
        
        if abs(distance - prev_distance) < 0.05:
            no_progress_count += 1
        else:
            no_progress_count = 0
            stuck_count = max(0, stuck_count - 1)
        
        if no_progress_count > 10:
            print(f"[Agent {agent_id}] No progress, trying alternative approach")
            import random
            for _ in range(3):
                safe_step(agent_id, {"action": "RotateRight", "degrees": random.choice([45, 90, 135])}, verbose=False)
                if safe_step(agent_id, {"action": "MoveAhead"}, verbose=False):
                    no_progress_count = 0
                    break
        
        if stuck_count > 4:
            import random
            rotation_angle = random.choice([45, -45, 90, -90, 135, -135])
            safe_step(agent_id, {"action": "RotateRight" if rotation_angle > 0 else "RotateLeft", 
                                "degrees": abs(rotation_angle)}, verbose=False)
            for _ in range(3):
                if safe_step(agent_id, {"action": "MoveAhead"}, verbose=False):
                    stuck_count = 0
                    break
            
            if stuck_count > 8:
                print(f"[Agent {agent_id}] Too many obstacles, cannot reach {object_name}")
                break
        
        prev_distance = distance
        rotate_to_face_target(agent_id, target_pos)
        moved = safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
        
        if not moved:
            stuck_count += 1
        else:
            stuck_count = max(0, stuck_count - 1)
    
    final_distance = calculate_distance(get_agent_position(agent_id), target_pos)
    return final_distance <= required_distance + 0.5

def safe_step(agent_id: int, action_dict: Dict, max_retries: int = 3, verbose: bool = True) -> bool:
    for attempt in range(max_retries):
        try:
            action_dict["agentId"] = agent_id
            with controller_lock:
                ev = controller.step(action_dict)
                if not ev or not ev.events or agent_id >= len(ev.events):
                    return False
                agent_event = ev.events[agent_id]
                success = agent_event.metadata["lastActionSuccess"]
            
            if success:
                return True
            else:
                if verbose and attempt == max_retries - 1:
                    error_msg = agent_event.metadata.get("errorMessage", "Unknown")
                    if action_dict["action"] not in ["MoveAhead"]:
                        print(f"[Agent {agent_id}] {action_dict['action']} failed: {error_msg[:60]}")
        except Exception as e:
            if verbose and attempt == max_retries - 1:
                print(f"[Agent {agent_id}] Exception: {str(e)[:60]}")
            if attempt < max_retries - 1:
                time.sleep(0.15)
    return False

def execute_single_task(agent_id: int, st: Dict) -> bool:
    """단일 작업 실행 - A* 네비게이션 재시도 로직"""
    action = st["action"]
    obj_name = st.get("object")
    target_name = st.get("target")
    required_dist = st.get("required_distance", 1.5)
    
    success = False
    max_retries = 3
    
    for attempt in range(max_retries):
        if attempt > 0:
            print(f"[Agent {agent_id}] Retrying {action} (attempt {attempt + 1}/{max_retries})")
            time.sleep(0.3)
        
        if action == "GotoObject":
            obj = find_object_in_scene(obj_name)
            if obj:
                target_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                success = navigate_to_position_astar(agent_id, target_pos, obj_name)
                
                if success:
                    obj_id = obj["objectId"]
                    if not is_object_visible(agent_id, obj_id):
                        look_for_object(agent_id, obj_id)
                    
                    if target_name or obj_name in ["Bowl", "Fridge", "Microwave"]:
                        for _ in range(3):
                            current_dist = calculate_distance(get_agent_position(agent_id), target_pos)
                            if current_dist < 1.0:
                                break
                            rotate_to_face_target(agent_id, target_pos)
                            safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
                    break
            else:
                print(f"[Agent {agent_id}] Object '{obj_name}' not found")
        
        elif action == "PickupObject":
            obj = find_object_in_scene(obj_name)
            if obj and obj.get("pickupable", False):
                object_id = obj["objectId"]
                obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
		
                rotate_to_face_target(agent_id, obj_pos)
                safe_step(agent_id, {"action": "LookDown", "degrees": 30}, verbose=False)
                
                if not is_object_visible(agent_id, object_id):
                    look_for_object(agent_id, object_id, max_rotations=2)
		    
                print(f"[Agent {agent_id}] Picking up {obj_name}...")
                success = safe_step(agent_id, {"action": "PickupObject", "objectId": object_id})
		
                if success:
                    ROBOTS[agent_id]["holding"] = obj_name
                    safe_step(agent_id, {"action": "LookUp", "degrees": 30}, verbose=False)
                    break
        
        elif action == "PutObject":
            target = find_object_in_scene(target_name)
            if target:
                target_id = target["objectId"]
                target_pos = (target["position"]["x"], target["position"]["y"], target["position"]["z"])
		
                rotate_to_face_target(agent_id, target_pos)
                safe_step(agent_id, {"action": "LookDown", "degrees": 30}, verbose=False)
		
                current_pos = get_agent_position(agent_id)
                dist = calculate_distance(current_pos, target_pos)
                if dist > 1.1:
                    print(f"[Agent {agent_id}] Target too far ({dist:.2f}m), moving closer...")
                    safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
                    rotate_to_face_target(agent_id, target_pos)

                if not is_object_visible(agent_id, target_id):
                    look_for_object(agent_id, target_id, max_rotations=2)
		        
                print(f"[Agent {agent_id}] Putting object in {target_name}...")
                success = safe_step(agent_id, {"action": "PutObject", "objectId": target_id})
		
                if success:
                    ROBOTS[agent_id]["holding"] = None
                    safe_step(agent_id, {"action": "LookUp", "degrees": 30}, verbose=False)
                    break
                else:
                    safe_step(agent_id, {"action": "RotateRight", "degrees": 10}, verbose=False)
        
        elif action == "OpenObject":
            obj = find_object_in_scene(obj_name)
            if obj and obj.get("openable", False):
                object_id = obj["objectId"]
                obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
        
                rotate_to_face_target(agent_id, obj_pos)
        
                if not is_object_visible(agent_id, object_id):
                    safe_step(agent_id, {"action": "LookDown", "degrees": 30}, verbose=False)
            
                    if not is_object_visible(agent_id, object_id):
                        navigate_to_position_astar(agent_id, obj_pos, obj_name)
                        look_for_object(agent_id, object_id, max_rotations=2)
        
                print(f"[Agent {agent_id}] Opening {obj_name}...")
                success = safe_step(agent_id, {"action": "OpenObject", "objectId": object_id})
        
                if success:
                    safe_step(agent_id, {"action": "LookUp", "degrees": 30}, verbose=False)
                    break
                else:
                    print(f"[Agent {agent_id}] Open failed, stepping back and retrying...")
                    safe_step(agent_id, {"action": "MoveBack"}, verbose=False)
                    rotate_to_face_target(agent_id, obj_pos)
        
        elif action == "CloseObject":
            obj = find_object_in_scene(obj_name)
            if obj and obj.get("openable", False):
                object_id = obj["objectId"]
                
                if not is_object_visible(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position_astar(agent_id, obj_pos, obj_name)
                    look_for_object(agent_id, object_id)
                
                print(f"[Agent {agent_id}] Closing {obj_name}...")
                success = safe_step(agent_id, {"action": "CloseObject", "objectId": object_id})
                if success:
                    break
        
        elif action == "ToggleObjectOn":
            obj = find_object_in_scene(obj_name)
            if obj and obj.get("toggleable", False):
                object_id = obj["objectId"]
                
                if not is_object_visible(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position_astar(agent_id, obj_pos, obj_name)
                    look_for_object(agent_id, object_id)
                
                print(f"[Agent {agent_id}] Toggling on {obj_name}...")
                success = safe_step(agent_id, {"action": "ToggleObjectOn", "objectId": object_id})
                if success:
                    break
        
        elif action == "ToggleObjectOff":
            obj = find_object_in_scene(obj_name)
            if obj and obj.get("toggleable", False):
                object_id = obj["objectId"]
                
                if not is_object_visible(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position_astar(agent_id, obj_pos, obj_name)
                    look_for_object(agent_id, object_id)
                
                print(f"[Agent {agent_id}] Toggling off {obj_name}...")
                success = safe_step(agent_id, {"action": "ToggleObjectOff", "objectId": object_id})
                if success:
                    break
    
    return success

def execute_agent_task_thread(agent_id: int, task: Dict, results: Dict, index: int, lock: threading.Lock):
    try:
        action = task["action"]
        obj_name = task.get("object") or task.get("target")
        
        with lock:
            print(f"\n[Task {task['task_id']}] Agent {agent_id} STARTED: {action} on {obj_name}")
            ROBOTS[agent_id]["status"] = "busy"
        
        success = execute_single_task(agent_id, task)
        
        with lock:
            ROBOTS[agent_id]["status"] = "idle"
            print(f"  [Task {task['task_id']}] Agent {agent_id} {'SUCCESS' if success else 'FAILED'}")
        
        results[index] = (task, success)
    except Exception as e:
        with lock:
            print(f"[Task {task['task_id']}] Agent {agent_id} Exception: {str(e)[:80]}")
            ROBOTS[agent_id]["status"] = "idle"
        results[index] = (task, False)

def execute_tasks_parallel(allocation: List[Tuple[int, Dict]], dependencies: Dict[int, List[int]]):
    completed = []
    failed = []
    completed_task_ids: Set[int] = set()
    failed_task_ids: Set[int] = set()
    
    print(f"\n{'='*60}")
    print(f"Starting parallel execution ({len(allocation)} tasks)")
    print(f"{'='*60}\n")
    
    task_dict = {task["task_id"]: (agent_id, task) for agent_id, task in allocation}
    print_lock = threading.Lock()
    
    max_iterations = len(allocation) + 5
    iteration = 0
    
    while len(completed_task_ids) + len(failed_task_ids) < len(allocation) and iteration < max_iterations:
        iteration += 1
        print(f"\n[Iteration {iteration}] Completed: {len(completed_task_ids)}, Failed: {len(failed_task_ids)}/{len(allocation)}")
        
        ready_tasks = []
        for task_id, (agent_id, task) in task_dict.items():
            if task_id in completed_task_ids or task_id in failed_task_ids:
                continue
            
            deps = dependencies.get(task_id, [])
            deps_satisfied = all(dep_id in completed_task_ids for dep_id in deps)
            
            if deps_satisfied:
                ready_tasks.append((agent_id, task))
        
        if not ready_tasks:
            remaining_tasks = len(allocation) - len(completed_task_ids) - len(failed_task_ids)
            if remaining_tasks > 0:
                print(f"[Warning] {remaining_tasks} tasks remain but dependencies not satisfied")
            break
        
        agents_in_use = set()
        parallel_batch = []
        
        for agent_id, task in ready_tasks:
            if agent_id not in agents_in_use and ROBOTS[agent_id]["status"] == "idle":
                parallel_batch.append((agent_id, task))
                agents_in_use.add(agent_id)
        
        if not parallel_batch:
            print("[Warning] No agents available, waiting...")
            time.sleep(1.0)
            continue
        
        print(f"Executing batch: {[(a, t['task_id'], t['action']) for a, t in parallel_batch]}")
        
        threads = []
        results = {}
        
        for idx, (agent_id, task) in enumerate(parallel_batch):
            thread = threading.Thread(
                target=execute_agent_task_thread,
                args=(agent_id, task, results, idx, print_lock),
                daemon=True
            )
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join(timeout=60.0)
        
        for idx in range(len(parallel_batch)):
            if idx in results:
                task, success = results[idx]
                
                if success:
                    completed.append(task)
                    completed_task_ids.add(task["task_id"])
                else:
                    failed.append(task)
                    failed_task_ids.add(task["task_id"])
        
        time.sleep(0.5)
    
    return completed, failed

# ----------------------------
# Scene 정보 가져오기
# ----------------------------
def get_scene_objects() -> List[str]:
    """Scene의 모든 객체 타입 반환"""
    ev = controller.last_event
    object_types = set()
    for obj in ev.metadata["objects"]:
        object_types.add(obj["objectType"])
    return sorted(list(object_types))

def spread_agents_initial_positions(agent_count: int):
    """에이전트들을 서로 다른 위치에 배치"""
    print(f"Spreading {agent_count} agents to separate locations...")
    
    if agent_count >= 2:
        controller.step({"action": "RotateRight", "degrees": 180, "agentId": 1})
        for _ in range(15):
            success = controller.step({"action": "MoveAhead", "agentId": 1})
            if not success.events[1].metadata["lastActionSuccess"]:
                break
        pos = controller.last_event.events[1].metadata["agent"]["position"]
        print(f"  Agent 1: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    if agent_count >= 3:
        controller.step({"action": "RotateRight", "degrees": 90, "agentId": 2})
        for _ in range(15):
            success = controller.step({"action": "MoveAhead", "agentId": 2})
            if not success.events[2].metadata["lastActionSuccess"]:
                break
        pos = controller.last_event.events[2].metadata["agent"]["position"]
        print(f"  Agent 2: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    if agent_count >= 4:
        controller.step({"action": "RotateLeft", "degrees": 90, "agentId": 3})
        for _ in range(15):
            success = controller.step({"action": "MoveAhead", "agentId": 3})
            if not success.events[3].metadata["lastActionSuccess"]:
                break
        pos = controller.last_event.events[3].metadata["agent"]["position"]
        print(f"  Agent 3: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    pos = controller.last_event.events[0].metadata["agent"]["position"]
    print(f"  Agent 0: ({pos['x']:.2f}, {pos['z']:.2f})")
    print("Agents positioned successfully\n")

# ----------------------------
# All Agent view 설정
# ----------------------------
stop_event = threading.Event()

def show_all_agent_views(agent_count: int):
    window_name = "All Agent Views"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 400 * agent_count, 400)
    
    while not stop_event.is_set():
        try:
            ev = controller.last_event
            if ev and len(ev.events) >= agent_count:
                all_frames = []
                colors = [(0, 255, 0), (255, 100, 0), (0, 100, 255), (255, 0, 255)] 

                for agent_id in range(agent_count):
                    frame = ev.events[agent_id].frame
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    agent_pos = get_agent_position(agent_id)
                    holding = ROBOTS[agent_id]["holding"]
                    status = ROBOTS[agent_id]["status"]
                    info_text = f"Agent {agent_id} ({status})"
                    info_text2 = f"Pos: ({agent_pos[0]:.1f}, {agent_pos[2]:.1f})"
                    info_text3 = f"Holding: {holding or 'None'}"
                    
                    color = colors[agent_id % len(colors)]
                    cv2.putText(frame_bgr, info_text, (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame_bgr, info_text2, (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame_bgr, info_text3, (10, 75),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    all_frames.append(frame_bgr)

                combined_frame = np.hstack(all_frames)
                cv2.imshow(window_name, combined_frame)

            if cv2.waitKey(100) & 0xFF == ord('q'):
                stop_event.set()
                break
        except Exception as e:
            break
    
    print("-> View thread is closing window and exiting.")
    cv2.destroyAllWindows()

# ----------------------------
# 메인 실행 로직
# ----------------------------
if __name__ == "__main__":
    # 사용자 입력 받기
    floor_plan, agent_count, high_level_task = get_user_input()
    
    print(f"\n{'='*60}")
    print(f"Initializing environment...")
    print(f"  FloorPlan: {floor_plan}")
    print(f"  Agents: {agent_count}")
    print(f"  Task: {high_level_task}")
    print(f"{'='*60}\n")
    
    # 환경 초기화
    try:
        controller = Controller(
            agentMode="default",
            visibilityDistance=1.5,
            renderDepthImage=False,
            renderInstanceSegmentation=False,
            scene=floor_plan,
            gridSize=0.25,
            agentCount=agent_count,
            snapToGrid=True,
            width=600,
            height=600
        )
        print("✓ Multi-agent environment initialized\n")
    except Exception as e:
        print(f"✗ Failed to initialize environment: {e}")
        exit(1)
    
    # A* Pathfinder 초기화
    pathfinder = AStarPathfinder(controller, grid_size=0.25)
    
    # 로봇 정의
    ROBOTS = [
        {"id": i, "name": f"robot{i}", 
         "skills": {"GotoObject", "PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"}, 
         "holding": None, "status": "idle"}
        for i in range(agent_count)
    ]
    
    # 에이전트 배치
    spread_agents_initial_positions(agent_count)
    
    # 뷰 스레드 시작
    view_thread = threading.Thread(target=show_all_agent_views, args=(agent_count,), daemon=True)
    view_thread.start()
    print("✓ All agent views window opened\n")
    
    # Scene 객체 가져오기
    scene_objects = get_scene_objects()
    print(f"Scene objects ({len(scene_objects)}): {', '.join(scene_objects[:15])}{'...' if len(scene_objects) > 15 else ''}\n")
    
    # LLM 기반 작업 분해
    print("Planning tasks with LLM...")
    subtasks = decompose_task_with_llm(high_level_task, scene_objects)
    
    if not subtasks:
        print("✗ Failed to decompose task. Exiting.")
        stop_event.set()
        controller.stop()
        exit(1)
    
    # LLM 기반 의존성 분석
    dependencies = analyze_dependencies_with_llm(subtasks)
    
    # LLM 기반 작업 할당
    allocation = allocate_tasks_with_llm(subtasks, ROBOTS)
    
    if not allocation:
        print("✗ Failed to allocate tasks. Exiting.")
        stop_event.set()
        controller.stop()
        exit(1)
    
    # 실행
    print("\n" + "="*60)
    print("Starting task execution...")
    print("="*60)
    
    start_time = time.time()
    completed_tasks, failed_tasks = execute_tasks_parallel(allocation, dependencies)
    execution_time = time.time() - start_time
    
    # 결과 출력
    print("\n" + "="*60)
    print("[EXECUTION SUMMARY]")
    print("="*60)
    total_tasks = len(allocation)
    success_rate = (len(completed_tasks) / total_tasks * 100) if total_tasks > 0 else 0
    
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Completed: {len(completed_tasks)} ({success_rate:.1f}%)")
    print(f"  Failed: {len(failed_tasks)}")
    print(f"  Execution Time: {execution_time:.2f}s")
    
    agent_stats = defaultdict(lambda: {"completed": 0, "failed": 0})
    completed_ids = {t["task_id"] for t in completed_tasks}
    for agent_id, task in allocation:
        status = "completed" if task["task_id"] in completed_ids else "failed"
        agent_stats[agent_id][status] += 1
    
    print("\n[AGENT PERFORMANCE]")
    for agent_id in sorted(agent_stats.keys()):
        stats = agent_stats[agent_id]
        total = stats['completed'] + stats['failed']
        rate = (stats['completed'] / total * 100) if total > 0 else 0
        print(f"  Agent {agent_id}: {stats['completed']}/{total} completed ({rate:.1f}%)")
    
    if failed_tasks:
        print("\n[FAILED TASKS]")
        for task in failed_tasks:
            obj = task.get('object') or task.get('target', '')
            print(f"  Task {task['task_id']}: {task['action']} - {obj}")
    
    print("="*60)
    
    # 종료
    print("\nPress Enter to exit...")
    input()
    
    print("\nShutting down...")
    stop_event.set()
    view_thread.join(timeout=2.0)
    controller.stop()
    print("✓ Cleanup complete")
