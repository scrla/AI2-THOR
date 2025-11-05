from ai2thor.controller import Controller
from collections import defaultdict
import time
import math
from typing import List, Dict, Optional, Tuple, Set
import cv2
import threading
import numpy as np

# ----------------------------
# 환경 초기화
# ----------------------------
AGENT_COUNT = 3
SCENE = "FloorPlan20_physics"

controller = Controller(
    agentMode="default",
    visibilityDistance=1.5,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    scene=SCENE,
    gridSize=0.25,
    agentCount=AGENT_COUNT,
    snapToGrid=True,
    width=800,
    height=800
)

print("Initialized multi-agent environment")

def spread_agents_initial_positions(controller, num_agents: int):
    """에이전트들을 다른 위치에 배치"""
    print(f"Spreading {num_agents} agents to separate locations...")
    
    positions = [
        {"agent_id": 0, "description": "Original position"},
        {"agent_id": 1, "description": "Opposite area)"},
        {"agent_id": 2, "description": "Another opposite area"},
    ]
    
    if num_agents >= 2:
        # Agent 1을 180도 회전 후 뒤로 이동 
        print(f"  Agent 1: Moving to opposite side...")
        controller.step({"action": "RotateRight", "degrees": 180, "agentId": 1})
        for _ in range(20):
            success = controller.step({"action": "MoveAhead", "agentId": 1})
            if not success.events[1].metadata["lastActionSuccess"]:
                break
        pos = controller.last_event.events[1].metadata["agent"]["position"]
        print(f"  Agent 1: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    if num_agents >= 3:
        # Agent 2를 90도 회전 후 앞으로 이동 
        print(f"  Agent 2: Moving to side area...")
        controller.step({"action": "RotateRight", "degrees": 90, "agentId": 2})
        for _ in range(20):
            success = controller.step({"action": "MoveAhead", "agentId": 2})
            if not success.events[2].metadata["lastActionSuccess"]:
                break
        pos = controller.last_event.events[2].metadata["agent"]["position"]
        print(f"  Agent 2: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    # Agent 0 위치 확인
    pos = controller.last_event.events[0].metadata["agent"]["position"]
    print(f"  Agent 0: ({pos['x']:.2f}, {pos['z']:.2f})")
    
    print("Agents positioned successfully\n")

spread_agents_initial_positions(controller, AGENT_COUNT)

# ----------------------------
# All Agent view 설정
# ----------------------------
stop_event = threading.Event() # 스레드 종료 Event 객체

def show_all_agent_views():
    """모든 에이전트의 시점을 하나의 창에 나란히 표시"""
    window_name = "All Agent Views"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600 * AGENT_COUNT, 600)
    
    # 종료 신호(stop_event) 전까지 계속 실행
    while not stop_event.is_set():
        try:
            ev = controller.last_event
            if ev and len(ev.events) >= AGENT_COUNT:
                all_frames = []
                colors = [(0, 255, 0), (255, 100, 0), (0, 100, 255)] 

                for agent_id in range(AGENT_COUNT):
                    frame = ev.events[agent_id].frame
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    agent_pos = get_agent_position(agent_id)
                    holding = ROBOTS[agent_id]["holding"]
                    info_text = f"Agent {agent_id} | Pos: ({agent_pos[0]:.1f}, {agent_pos[2]:.1f}) | Holding: {holding or 'None'}"
                    
                    color = colors[agent_id % len(colors)]
                    cv2.putText(frame_bgr, info_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    all_frames.append(frame_bgr)

                # 모든 프레임 수평으로 합침
                combined_frame = np.hstack(all_frames)
                cv2.imshow(window_name, combined_frame)

            # waitKey를 루프마다 호출하여 창이 멈추는 현상 방지
            if cv2.waitKey(100) & 0xFF == ord('q'):
                stop_event.set()
                break
        except Exception:
            break
            
    print("-> View thread is closing window and exiting.")
    cv2.destroyAllWindows()


# 뷰 스레드 지정
view_thread = threading.Thread(target=show_all_agent_views, daemon=True)
view_thread.start()
print("All agent views window opened")

# ----------------------------
# 로봇 상태 정의
# ----------------------------
ROBOTS = [
    {"id": 0, "name": "robot0", "skills": {"GotoObject", "PickupObject", "PutObject"}, "holding": None, "status": "idle"},
    {"id": 1, "name": "robot1", "skills": {"GotoObject", "OpenObject", "CloseObject", "ToggleObjectOn"}, "holding": None, "status": "idle"},
    {"id": 2, "name": "robot2", "skills": {"GotoObject", "PickupObject", "PutObject", "OpenObject", "CloseObject"}, "holding": None, "status": "idle"},
]

# ----------------------------
# 작업 정의
# ----------------------------
HIGH_LEVEL_TASK = "put the apple in the bowl and switch on the microwave and put the tomato in the pot"

def decompose_task(task_text):
    """작업을 subtasks로 분해"""
    subtasks = []
    t = task_text.lower()
    
    if "apple" in t and "bowl" in t:
        subtasks += [
            {"action": "GotoObject", "object": "Apple", "agent_pref": [0], "required_distance": 1.2, "chain": "apple"},
            {"action": "PickupObject", "object": "Apple", "agent_pref": [0], "chain": "apple"},
            {"action": "GotoObject", "object": "Bowl", "agent_pref": [0], "required_distance": 1.0, "chain": "apple"},
            {"action": "PutObject", "target": "Bowl", "agent_pref": [0], "chain": "apple"}
        ]
    
    if "tomato" in t and "pot" in t:
        subtasks += [
            {"action": "GotoObject", "object": "Tomato", "agent_pref": [2], "required_distance": 1.2, "chain": "tomato_pot"},
            {"action": "PickupObject", "object": "Tomato", "agent_pref": [2], "chain": "tomato_pot"},
            {"action": "GotoObject", "object": "Pot", "agent_pref": [2], "required_distance": 1.0, "chain": "tomato_pot"},
            {"action": "PutObject", "target": "Pot", "agent_pref": [2], "chain": "tomato_pot"}
        ]
    
    if "tomato" in t and "refrigerator" in t:
        subtasks += [
            {"action": "GotoObject", "object": "Tomato", "agent_pref": [2], "required_distance": 1.2, "chain": "tomato_fridge"},
            {"action": "PickupObject", "object": "Tomato", "agent_pref": [2], "chain": "tomato_fridge"},
            {"action": "GotoObject", "object": "Fridge", "agent_pref": [2], "required_distance": 1.2, "chain": "tomato_fridge"},
            {"action": "OpenObject", "object": "Fridge", "agent_pref": [2], "chain": "tomato_fridge"},
            {"action": "PutObject", "target": "Fridge", "agent_pref": [2], "chain": "tomato_fridge"},
            {"action": "CloseObject", "object": "Fridge", "agent_pref": [2], "chain": "tomato_fridge"}
        ]
        
    if "switch on" in t or "turn on" in t:
        subtasks += [
            {"action": "GotoObject", "object": "Microwave", "agent_pref": [1], "required_distance": 1.2, "chain": "microwave"},
            {"action": "ToggleObjectOn", "object": "Microwave", "agent_pref": [1], "chain": "microwave"}
        ]
    
    return subtasks

subtasks = decompose_task(HIGH_LEVEL_TASK)
print(f"Decomposed {len(subtasks)} subtasks")

# ----------------------------
# 유틸리티 함수들
# ----------------------------
def get_agent_position(agent_id: int) -> Tuple[float, float, float]:
    """Agents의 현재 위치 반환"""
    ev = controller.last_event
    agent_meta = ev.events[agent_id].metadata
    pos = agent_meta["agent"]["position"]
    return (pos["x"], pos["y"], pos["z"])

def get_agent_rotation(agent_id: int) -> float:
    """Agents의 현재 회전 각도 반환"""
    ev = controller.last_event
    agent_meta = ev.events[agent_id].metadata
    return agent_meta["agent"]["rotation"]["y"]

def calculate_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """두 위치 간 유클리드 거리 계산"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[2] - pos2[2])**2)

def find_object_in_scene(object_name: str) -> Optional[Dict]:
    """현재 scene에서 객체 찾기"""
    ev = controller.last_event
    
    for o in ev.metadata["objects"]:
        if object_name.lower() == o["objectType"].lower():
            return o
        if object_name.lower() in o["objectType"].lower():
            return o
    
    return None

def get_visible_objects(agent_id: int) -> List[str]:
    """Agents가 볼 수 있는 객체 ID 리스트"""
    ev = controller.last_event
    agent_event = ev.events[agent_id]
    return [o["objectId"] for o in agent_event.metadata["objects"] if o.get("visible", False)]

def is_object_visible(agent_id: int, object_id: str) -> bool:
    """Agents가 특정 객체를 볼 수 있는지 확인"""
    return object_id in get_visible_objects(agent_id)

def look_for_object(agent_id: int, object_id: str, max_rotations: int = 8) -> bool:
    """360도 회전하며 객체 찾기"""
    for i in range(max_rotations):
        if is_object_visible(agent_id, object_id):
            return True
        safe_step(agent_id, {"action": "RotateRight", "degrees": 45}, verbose=False)
    return False

def calculate_angle_to_target(agent_pos: Tuple[float, float, float], 
                               agent_rotation: float,
                               target_pos: Tuple[float, float, float]) -> float:
    """타겟까지의 상대 각도 계산"""
    dx = target_pos[0] - agent_pos[0]
    dz = target_pos[2] - agent_pos[2]
    
    target_angle = math.degrees(math.atan2(dx, dz))
    relative_angle = (target_angle - agent_rotation + 180) % 360 - 180
    
    return relative_angle

# ----------------------------
# 작업 할당
# ----------------------------
def allocate_tasks(subtasks: List[Dict], robots: List[Dict]) -> List[Tuple[int, Dict]]:
    """작업 할당 - agent_pref를 우선"""
    allocation = []
    
    for idx, st in enumerate(subtasks):
        action = st["action"]
        preferred_agents = st.get("agent_pref", [])
        
        if preferred_agents:
            candidates = [r for r in robots if r["id"] in preferred_agents]
            if candidates:
                chosen = candidates[0]
            else:
                print(f"Warning: Preferred agent {preferred_agents} not found for task {idx}")
                chosen = robots[0]
        else:
            candidates = [r for r in robots if action in r["skills"]]
            chosen = candidates[0] if candidates else robots[0]
        
        st["task_id"] = idx
        allocation.append((chosen["id"], st))
    
    return allocation

allocation = allocate_tasks(subtasks, ROBOTS)
print("\nTask allocation:")
for agent_id, task in allocation:
    print(f"  Task {task['task_id']}: Agent {agent_id} -> {task['action']}")

# ----------------------------
# 네비게이션 함수
# ----------------------------
def rotate_to_face_target(agent_id: int, target_pos: Tuple[float, float, float]) -> bool:
    """타겟을 향해 회전"""
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

def find_passable_direction(agent_id: int, max_attempts: int = 8) -> bool:
    """막혔을 때 통과 가능한 방향 찾기"""
    print(f"[Agent {agent_id}] Finding passable direction...")
    original_rotation = get_agent_rotation(agent_id)
    
    for direction in range(max_attempts):
        rotation_angle = 45 * direction
        
        safe_step(agent_id, {"action": "RotateLeft", "degrees": rotation_angle}, verbose=False)
        
        test_event = controller.last_event
        test_moved = safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
        
        if test_moved:
            print(f"[Agent {agent_id}] Found passable direction at {rotation_angle}°")
            return True
        else:
            restore_angle = rotation_angle
            if restore_angle > 0:
                safe_step(agent_id, {"action": "RotateRight", "degrees": restore_angle}, verbose=False)
    
    print(f"[Agent {agent_id}] No passable direction found")
    return False

def navigate_to_position(agent_id: int, target_pos: Tuple[float, float, float], 
                        object_name: str = "target", 
                        required_distance: float = 1.5,
                        max_steps: int = 40) -> bool:
    """위치 기반 경로 찾기"""
    print(f"[Agent {agent_id}] Navigating to {object_name} (target: {required_distance}m)...")
    
    stuck_count = 0
    prev_distance = float('inf')
    no_progress_count = 0
    
    for step in range(max_steps):
        agent_pos = get_agent_position(agent_id)
        distance = calculate_distance(agent_pos, target_pos)
        
        # 목표 거리 도달
        if distance <= required_distance:
            print(f"[Agent {agent_id}] Reached {object_name} (distance: {distance:.2f}m)")
            return True
        
        # 진행 상황 확인
        if abs(distance - prev_distance) < 0.05:
            no_progress_count += 1
        else:
            no_progress_count = 0
            stuck_count = 0
        
        # 진행이 없으면 조기 종료
        if no_progress_count > 8:
            print(f"[Agent {agent_id}] No progress for 8 steps, stopping navigation")
            break
        
        # 막힌 경우 우회 시도
        if stuck_count > 3:
            print(f"[Agent {agent_id}] Detour attempt {stuck_count}")
            
            # 랜덤 방향으로 회전 후 이동
            import random
            rotation_angle = random.choice([45, -45, 90, -90])
            safe_step(agent_id, {"action": "RotateRight" if rotation_angle > 0 else "RotateLeft", 
                                "degrees": abs(rotation_angle)}, verbose=False)
            
            for _ in range(2):
                if safe_step(agent_id, {"action": "MoveAhead"}, verbose=False):
                    stuck_count = 0
                    break
            
            # 3번 우회 실패 시 포기
            if stuck_count > 6:
                print(f"[Agent {agent_id}] Too many detour attempts, giving up")
                break
        
        prev_distance = distance
        
        # 타겟 방향으로 회전
        rotate_to_face_target(agent_id, target_pos)
        
        # 전진 시도
        moved = safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
        
        if not moved:
            stuck_count += 1
        else:
            stuck_count = max(0, stuck_count - 1)  # 성공 시 stuck_count 감소
    
    # 최종 거리 체크
    final_distance = calculate_distance(get_agent_position(agent_id), target_pos)
    tolerance = 0.5
    
    if final_distance <= required_distance + tolerance:
        print(f"[Agent {agent_id}] Close enough to {object_name} ({final_distance:.2f}m)")
        return True
    
    print(f"[Agent {agent_id}]  Could not reach {object_name} ({final_distance:.2f}m)")
    return False
    
# ----------------------------
# 실행 함수
# ----------------------------
controller_lock = threading.Lock()

def safe_step(agent_id: int, action_dict: Dict, max_retries: int = 1, verbose: bool = True) -> bool:
    """액션 실행"""
    for attempt in range(max_retries):
        try:
            action_dict["agentId"] = agent_id
            
            # Lock으로 controller 동시 접근 방지
            with controller_lock:
                ev = controller.step(action_dict)
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
                print(f"[Agent {agent_id}]  Exception: {str(e)[:60]}")
            time.sleep(0.1)  # 재시도 전 대기
    
    return False

def execute_single_task(agent_id: int, st: Dict) -> bool:
    """단일 작업 실행"""
    action = st["action"]
    obj_name = st.get("object")
    target_name = st.get("target")
    required_dist = st.get("required_distance", 1.5)
    
    success = False
    
    if action == "GotoObject":
        obj = find_object_in_scene(obj_name)
        if obj:
            target_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
            success = navigate_to_position(agent_id, target_pos, obj_name, required_dist)
            
            if success:
                obj_id = obj["objectId"]
                if not is_object_visible(agent_id, obj_id):
                    look_for_object(agent_id, obj_id)
                
                if target_name or obj_name in ["Bowl", "Fridge"]:
                    print(f"[Agent {agent_id}] Getting closer for interaction...")
                    for _ in range(3):
                        current_dist = calculate_distance(get_agent_position(agent_id), target_pos)
                        if current_dist < 1.0:
                            break
                        rotate_to_face_target(agent_id, target_pos)
                        safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
        else:
            print(f"[Agent {agent_id}] Object '{obj_name}' not found")
    
    elif action == "PickupObject":
        obj = find_object_in_scene(obj_name)
        if obj and obj.get("pickupable", False):
            object_id = obj["objectId"]
            
            if not is_object_visible(agent_id, object_id):
                print(f"[Agent {agent_id}] Looking for {obj_name}...")
                if not look_for_object(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position(agent_id, obj_pos, obj_name, 1.0, max_steps=15)
                    look_for_object(agent_id, object_id)
            
            print(f"[Agent {agent_id}] Picking up {obj_name}...")
            success = safe_step(agent_id, {"action": "PickupObject", "objectId": object_id})
            if success:
                ROBOTS[agent_id]["holding"] = obj_name
        else:
            print(f"[Agent {agent_id}] Cannot pickup {obj_name}")
    
    elif action == "PutObject":
        target = find_object_in_scene(target_name)
        if target:
            target_id = target["objectId"]
            
            if not is_object_visible(agent_id, target_id):
                print(f"[Agent {agent_id}] Looking for {target_name}...")
                if not look_for_object(agent_id, target_id):
                    target_pos = (target["position"]["x"], target["position"]["y"], target["position"]["z"])
                    navigate_to_position(agent_id, target_pos, target_name, 0.8, max_steps=15)
                    look_for_object(agent_id, target_id)
            
            if not is_object_visible(agent_id, target_id):
                print(f"[Agent {agent_id}] Target not visible, adjusting position...")
                target_pos = (target["position"]["x"], target["position"]["y"], target["position"]["z"])
                rotate_to_face_target(agent_id, target_pos)
                for _ in range(2):
                    safe_step(agent_id, {"action": "MoveAhead"}, verbose=False)
            
            print(f"[Agent {agent_id}] Putting object in {target_name}...")
            
            target_pos = (target["position"]["x"], target["position"]["y"], target["position"]["z"])
            current_dist = calculate_distance(get_agent_position(agent_id), target_pos)
            print(f"[Agent {agent_id}] Current distance to {target_name}: {current_dist:.2f}m")
            
            success = safe_step(agent_id, {"action": "PutObject", "objectId": target_id})
            if success:
                ROBOTS[agent_id]["holding"] = None
        else:
            print(f"[Agent {agent_id}] Target '{target_name}' not found")
    
    elif action == "OpenObject":
        obj = find_object_in_scene(obj_name)
        if obj and obj.get("openable", False):
            object_id = obj["objectId"]
            
            if not is_object_visible(agent_id, object_id):
                print(f"[Agent {agent_id}] Looking for {obj_name}...")
                if not look_for_object(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position(agent_id, obj_pos, obj_name, 1.2, max_steps=15)
                    look_for_object(agent_id, object_id)
            
            print(f"[Agent {agent_id}] Opening {obj_name}...")
            success = safe_step(agent_id, {"action": "OpenObject", "objectId": object_id})
        else:
            print(f"[Agent {agent_id}] Cannot open {obj_name}")
    
    elif action == "CloseObject":
        obj = find_object_in_scene(obj_name)
        if obj and obj.get("openable", False):
            object_id = obj["objectId"]
            
            if not is_object_visible(agent_id, object_id):
                print(f"[Agent {agent_id}] Looking for {obj_name}...")
                look_for_object(agent_id, object_id)
            
            print(f"[Agent {agent_id}] Closing {obj_name}...")
            success = safe_step(agent_id, {"action": "CloseObject", "objectId": object_id})
        else:
            print(f"[Agent {agent_id}] Cannot close {obj_name}")
    
    elif action == "ToggleObjectOn":
        obj = find_object_in_scene(obj_name)
        if obj and obj.get("toggleable", False):
            object_id = obj["objectId"]
            
            if not is_object_visible(agent_id, object_id):
                print(f"[Agent {agent_id}] Looking for {obj_name}...")
                if not look_for_object(agent_id, object_id):
                    obj_pos = (obj["position"]["x"], obj["position"]["y"], obj["position"]["z"])
                    navigate_to_position(agent_id, obj_pos, obj_name, 1.2, max_steps=15)
                    look_for_object(agent_id, object_id)
            
            print(f"[Agent {agent_id}] Toggling on {obj_name}...")
            success = safe_step(agent_id, {"action": "ToggleObjectOn", "objectId": object_id})
        else:
            print(f"[Agent {agent_id}] Cannot toggle {obj_name}")
    
    return success

# ----------------------------
# 병렬 실행 로직
# ----------------------------
def build_dependency(allocation: List[Tuple[int, Dict]]) -> Dict[int, List[int]]:
    """작업 의존성 생성 (같은 chain의 작업은 순차적으로 실행)"""
    dependencies = defaultdict(list)
    chain_last_task = {}
    
    for agent_id, task in allocation:
        task_id = task["task_id"]
        chain = task.get("chain", None)
        
        if chain:
            # 같은 체인의 이전 작업에 의존
            if chain in chain_last_task:
                dependencies[task_id].append(chain_last_task[chain])
            chain_last_task[chain] = task_id
    
    return dependencies

def execute_agent_task_thread(agent_id: int, task: Dict, results: Dict, index: int, lock: threading.Lock):
    """스레드에서 실행될 에이전트 작업"""
    try:
        action = task["action"]
        obj_name = task.get("object") or task.get("target")
        
        with lock:
            print(f"\n[Task {task['task_id']}] Agent {agent_id} STARTED: {action} on {obj_name}")
            ROBOTS[agent_id]["status"] = "busy"
        
        success = execute_single_task(agent_id, task)
        
        with lock:
            ROBOTS[agent_id]["status"] = "idle"
            if success:
                print(f"  [Task {task['task_id']}] Agent {agent_id} SUCCESS")
            else:
                print(f"  [Task {task['task_id']}] Agent {agent_id} FAILED")
        
        results[index] = (task, success)
        
    except Exception as e:
        with lock:
            print(f"[Task {task['task_id']}] Agent {agent_id} Exception: {str(e)[:80]}")
        results[index] = (task, False)

def execute_tasks_parallel(allocation: List[Tuple[int, Dict]]):
    """병렬 작업 실행 (멀티스레딩)"""
    completed = []
    failed = []
    completed_task_ids: Set[int] = set()
    
    # 의존성 설정
    dependencies = build_dependency(allocation)
    
    print(f"\n{'='*60}")
    print(f"Starting TRUE PARALLEL execution ({len(allocation)} tasks)")
    print(f"Using multithreading for simultaneous execution")
    print(f"{'='*60}\n")
    
    # 의존성 정보 출력
    print("Task dependencies:")
    for agent_id, task in allocation:
        task_id = task["task_id"]
        deps = dependencies.get(task_id, [])
        if deps:
            print(f"  Task {task_id}: depends on {deps}")
        else:
            print(f"  Task {task_id}: independent")
    print()
    
    # 작업 인덱스 생성
    task_dict = {task["task_id"]: (agent_id, task) for agent_id, task in allocation}
    
    # 스레드 동기화용 lock 
    print_lock = threading.Lock()
    
    max_iterations = len(allocation) + 2
    iteration = 0
    
    # 실패한 작업 추적
    failed_task_ids: Set[int] = set()
    
    while len(completed_task_ids) < len(allocation) and iteration < max_iterations:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"[Iteration {iteration}] Completed: {len(completed_task_ids)}/{len(allocation)}")
        print(f"{'='*60}")
        
        # 실행 가능 작업 찾기
        ready_tasks = []
        for task_id, (agent_id, task) in task_dict.items():
            if task_id in completed_task_ids or task_id in failed_task_ids:
                continue
            
            # 의존성 체크
            deps = dependencies.get(task_id, [])
            if all(dep_id in completed_task_ids for dep_id in deps):
                ready_tasks.append((agent_id, task))
        
        if not ready_tasks:
            print("No ready tasks found")
            break
        
        print(f"Ready tasks: {[t[1]['task_id'] for t in ready_tasks]}")
        
        # 병렬 실행 작업 그룹화
        agents_in_use = set()
        parallel_batch = []
        
        for agent_id, task in ready_tasks:
            if agent_id not in agents_in_use:
                parallel_batch.append((agent_id, task))
                agents_in_use.add(agent_id)
        
        print(f"Executing SIMULTANEOUSLY: {[(a, t['task_id']) for a, t in parallel_batch]}")
        print(f"   {len(parallel_batch)} agents working in parallel\n")
        
        # 멀티스레딩
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
        
        # 전체 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        # 결과 수집
        for idx in range(len(parallel_batch)):
            if idx in results:
                task, success = results[idx]
                
                if success:
                    completed.append(task)
                    completed_task_ids.add(task["task_id"])
                else:
                    failed.append(task)
                    failed_task_ids.add(task["task_id"])  # 실패 추적
        
        # 동기화 대기
        time.sleep(0.5)
    
    return completed, failed

# ----------------------------
# 메인 실행 로직
# ----------------------------
if __name__ == "__main__":
    completed_tasks, failed_tasks = execute_tasks_parallel(allocation)

    print("\n" + "="*60)
    print("[Execution Summary]:")
    total_tasks = len(allocation)
    print(f"  Completed: {len(completed_tasks)}/{total_tasks} ({(len(completed_tasks)/total_tasks*100):.1f}%)")
    print(f"  Failed: {len(failed_tasks)}/{total_tasks}")

    agent_stats = defaultdict(lambda: {"completed": 0, "failed": 0})
    completed_ids = {t["task_id"] for t in completed_tasks}
    for agent_id, task in allocation:
        status = "completed" if task["task_id"] in completed_ids else "failed"
        agent_stats[agent_id][status] += 1

    for agent_id in sorted(agent_stats.keys()):
        stats = agent_stats[agent_id]
        print(f"  Agent {agent_id}: {stats['completed']} COMPLETED/ {stats['failed']} FAILED")
    print("="*60)

    # 종료
    print("\nShutting down...")
    
    # 뷰 스레드에게 종료 신호를보냄
    stop_event.set()
    
    # 뷰 스레드 종료 기다림
    view_thread.join(timeout=2.0)
    
    # 컨트롤러 종료
    controller.stop()
