"""
State Tracker - Tracks active containers and memory usage
Container Lifecycle Contract (4.3):
CREATED → STARTING → ALLOCATING_MEMORY → RUNNING → RELEASING_MEMORY → COMPLETED
                                              ↘ FAILED (OOM or timeout)
"""
import logging
import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ContainerState(Enum):
    """
    Container lifecycle states as per requirement 4.3
    Each state transition is logged, reported, and recoverable
    """
    CREATED = "created"              # Container registered but not started
    STARTING = "starting"            # Container process starting
    ALLOCATING_MEMORY = "allocating_memory"  # GPU memory allocation in progress
    RUNNING = "running"              # Container running with allocated memory
    RELEASING_MEMORY = "releasing_memory"    # Memory release in progress
    COMPLETED = "completed"          # Successfully completed
    FAILED = "failed"                # Failed (OOM or timeout)


class SystemState(Enum):
    """System-wide states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    SHUTDOWN = "shutdown"
    ERROR = "error"


@dataclass
class StateTransition:
    """Record of a state transition"""
    timestamp: str
    from_state: Optional[str]
    to_state: str
    message: str


@dataclass
class ContainerInfo:
    """Information about a running container"""
    container_id: int
    state: ContainerState
    memory_mb: float
    memory_block_id: Optional[int]
    process_id: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: int = 0
    state_transitions: List[StateTransition] = field(default_factory=list)
    failure_reason: Optional[str] = None


class StateTracker:
    """
    Tracks system state and active containers
    Implements Container Lifecycle Contract (4.3)
    """

    def __init__(self, max_concurrent_containers: int = 3):
        self.system_state = SystemState.INITIALIZING
        self.max_concurrent_containers = max_concurrent_containers
        self.containers: Dict[int, ContainerInfo] = {}
        self.container_counter = 0
        self.lock = threading.Lock()

        # Metrics
        self.metrics = {
            "containers_launched": 0,
            "containers_completed": 0,
            "containers_failed": 0,
            "containers_oom": 0,
            "peak_concurrent_containers": 0,
        }

        # OOM Tracking (Requirement 5.2)
        self.consecutive_oom_failures = 0
        self.oom_failure_threshold = 3  # Trigger reset after 3 consecutive

        self.event_log = []
        self.start_time = time.time()

    def register_container(
        self,
        memory_mb: float,
        memory_block_id: int,
        duration_seconds: int
    ) -> int:
        """
        Register a new container
        Initial state: CREATED
        Requirement 4.3: Logged with timestamp, reported to scheduler
        Returns container_id
        """
        with self.lock:
            self.container_counter += 1
            container_id = self.container_counter

            container = ContainerInfo(
                container_id=container_id,
                state=ContainerState.CREATED,
                memory_mb=memory_mb,
                memory_block_id=memory_block_id,
                start_time=time.time(),
                duration_seconds=duration_seconds
            )

            self.containers[container_id] = container
            self.metrics["containers_launched"] += 1

            # Log state transition (requirement 4.3)
            self._log_state_transition(
                container_id,
                None,
                ContainerState.CREATED,
                "Container registered"
            )

            current_running = self._count_running_containers()
            self.metrics["peak_concurrent_containers"] = max(
                self.metrics["peak_concurrent_containers"],
                current_running
            )

            logger.info(f"Container {container_id} CREATED: {memory_mb}MB, {duration_seconds}s")
            return container_id

    def update_container_state(
        self,
        container_id: int,
        new_state: ContainerState,
        process_id: Optional[int] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Update container state
        Requirement 4.3: Logged with timestamp, reported to scheduler, recoverable
        """
        with self.lock:
            if container_id not in self.containers:
                logger.warning(f"Container {container_id} not found")
                return False

            container = self.containers[container_id]
            old_state = container.state

            # Validate state transition
            if not self._is_valid_transition(old_state, new_state):
                logger.error(f"Invalid transition: {old_state.value} → {new_state.value}")
                return False

            container.state = new_state
            if process_id:
                container.process_id = process_id
            if reason:
                container.failure_reason = reason

            # Log state transition (requirement 4.3)
            self._log_state_transition(
                container_id,
                old_state,
                new_state,
                reason or f"State transition: {old_state.value} → {new_state.value}"
            )

            # Log based on state
            state_emoji = {
                ContainerState.CREATED: "[Registered]",
                ContainerState.STARTING: "[Starting]",
                ContainerState.ALLOCATING_MEMORY: "[Allocating memory]",
                ContainerState.RUNNING: "[Running]",
                ContainerState.RELEASING_MEMORY: "[Releasing memory]",
                ContainerState.COMPLETED: "[Completed]",
                ContainerState.FAILED: "[Failed]",
            }

            logger.info(
                f"{state_emoji.get(new_state, '')} Container {container_id}: "
                f"{old_state.value} → {new_state.value}"
            )

            return True

    def _is_valid_transition(self, from_state: ContainerState, to_state: ContainerState) -> bool:
        """Validate state transitions are allowed"""
        valid_transitions = {
            ContainerState.CREATED: [ContainerState.STARTING],
            ContainerState.STARTING: [ContainerState.ALLOCATING_MEMORY, ContainerState.FAILED],
            ContainerState.ALLOCATING_MEMORY: [ContainerState.RUNNING, ContainerState.FAILED],
            ContainerState.RUNNING: [ContainerState.RELEASING_MEMORY, ContainerState.FAILED],
            ContainerState.RELEASING_MEMORY: [ContainerState.COMPLETED, ContainerState.FAILED],
            ContainerState.COMPLETED: [],
            ContainerState.FAILED: [],
        }
        return to_state in valid_transitions.get(from_state, [])

    def mark_container_completed(self, container_id: int, success: bool = True, reason: Optional[str] = None):
        """
        Mark container as completed or failed
        Requirement 4.3: Logged with timestamp, reported to scheduler
        """
        with self.lock:
            if container_id not in self.containers:
                logger.warning(f"Container {container_id} not found")
                return

            container = self.containers[container_id]
            container.end_time = time.time()

            if success:
                new_state = ContainerState.COMPLETED
                self.metrics["containers_completed"] += 1
                logger.info(f"Container {container_id} COMPLETED")
            else:
                new_state = ContainerState.FAILED
                container.failure_reason = reason or "Unknown error"
                self.metrics["containers_failed"] += 1
                if reason and "OOM" in reason.upper():
                    self.metrics["containers_oom"] += 1
                logger.error(f"Container {container_id} FAILED: {reason}")

            # Update state
            old_state = container.state
            container.state = new_state

            # Log transition
            self._log_state_transition(
                container_id,
                old_state,
                new_state,
                f"Container completed: {reason}" if reason else "Container completed successfully"
            )

    def _log_state_transition(
        self,
        container_id: int,
        from_state: Optional[ContainerState],
        to_state: ContainerState,
        message: str
    ):
        """
        Log state transition with timestamp (requirement 4.3)
        Reported to scheduler via event log
        """
        timestamp = datetime.now().isoformat()

        event = {
            "timestamp": timestamp,
            "container_id": container_id,
            "from_state": from_state.value if from_state else None,
            "to_state": to_state.value,
            "message": message,
            "type": "state_transition"
        }

        self.event_log.append(event)

        # Also track in container's transition history
        if container_id in self.containers:
            container = self.containers[container_id]
            container.state_transitions.append(StateTransition(
                timestamp=timestamp,
                from_state=from_state.value if from_state else None,
                to_state=to_state.value,
                message=message
            ))

    def get_running_containers(self) -> Dict[int, ContainerInfo]:
        """Get all running containers"""
        with self.lock:
            return {
                cid: c for cid, c in self.containers.items()
                if c.state == ContainerState.RUNNING
            }

    def get_container_info(self, container_id: int) -> Optional[ContainerInfo]:
        """Get info about specific container"""
        with self.lock:
            return self.containers.get(container_id)

    def can_launch_container(self) -> bool:
        """Check if we can launch another container"""
        with self.lock:
            running_count = self._count_running_containers()
            return running_count < self.max_concurrent_containers

    def get_system_stats(self) -> dict:
        """Get system statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            running_count = self._count_running_containers()

            return {
                "system_state": self.system_state.value,
                "uptime_seconds": uptime,
                "total_containers_registered": self.container_counter,
                "containers_launched": self.metrics["containers_launched"],
                "containers_completed": self.metrics["containers_completed"],
                "containers_failed": self.metrics["containers_failed"],
                "containers_oom": self.metrics["containers_oom"],
                "running_containers": running_count,
                "peak_concurrent_containers": self.metrics["peak_concurrent_containers"],
                "max_concurrent_containers": self.max_concurrent_containers,
            }

    def set_system_state(self, state: SystemState):
        """Set system state"""
        with self.lock:
            old_state = self.system_state
            self.system_state = state
            self._log_event("system_state_changed", f"System state: {old_state.value} → {state.value}", {
                "old_state": old_state.value,
                "new_state": state.value
            })
            logger.info(f"System state: {old_state.value} → {state.value}")

    def _count_running_containers(self) -> int:
        """Count currently running containers (must be called with lock held)"""
        return sum(1 for c in self.containers.values() if c.state == ContainerState.RUNNING)

    def _log_event(self, event_type: str, message: str, data: dict):
        """Log an event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "message": message,
            "data": data
        }
        self.event_log.append(event)

    def get_event_log(self) -> list:
        """Get event log"""
        with self.lock:
            return list(self.event_log)

    def get_container_state_history(self, container_id: int) -> Optional[List[dict]]:
        """
        Get state transition history for a container (requirement 4.3)
        Useful for debugging and recovery
        """
        with self.lock:
            container = self.containers.get(container_id)
            if container:
                return [
                    {
                        "timestamp": t.timestamp,
                        "from_state": t.from_state,
                        "to_state": t.to_state,
                        "message": t.message
                    }
                    for t in container.state_transitions
                ]
            return None

    def get_failed_containers_requiring_recovery(self) -> List[dict]:
        """
        Get containers that failed during ALLOCATING_MEMORY (requirement 4.3 - recoverable)
        Scheduler should reclaim memory slots for these
        """
        with self.lock:
            failed = []
            for cid, container in self.containers.items():
                if container.state == ContainerState.FAILED:
                    # Check if failure happened during memory allocation
                    if container.state_transitions:
                        last_transition = container.state_transitions[-1]
                        if last_transition.from_state == ContainerState.ALLOCATING_MEMORY.value:
                            failed.append({
                                "container_id": cid,
                                "memory_mb": container.memory_mb,
                                "memory_block_id": container.memory_block_id,
                                "reason": container.failure_reason
                            })
            return failed

    def record_oom_event(self):
        """Record an Out-Of-Memory event"""
        self._log_event(
            "oom_event",
            "Out of memory: insufficient GPU memory for next container",
            {"timestamp": time.time()}
        )
        self.metrics["containers_oom"] += 1

    def get_last_successfully_launched_container(self) -> Optional[ContainerInfo]:
        """
        Get the last container that was successfully launched (reached RUNNING state or beyond)
        This is used to calculate memory for the next container in the exponential growth sequence
        Per requirement 5.2: Memory multiplier continues from last successful launch, not restarted on failures
        """
        with self.lock:
            # Find containers that reached at least RUNNING state (successful allocation + execution)
            successful_containers = [
                c for c in self.containers.values()
                if c.state in [
                    ContainerState.RUNNING,
                    ContainerState.RELEASING_MEMORY,
                    ContainerState.COMPLETED
                ]
            ]

            if not successful_containers:
                return None

            # Return the one with highest container_id (most recent successful launch)
            return max(successful_containers, key=lambda c: c.container_id)

    def increment_consecutive_oom_failures(self):
        """
        Increment consecutive OOM failure counter (Requirement 5.2)
        Used to detect when to trigger scheduler reset
        """
        with self.lock:
            self.consecutive_oom_failures += 1
            logger.warning(f"OOM Failure #{self.consecutive_oom_failures} - Threshold for reset: {self.oom_failure_threshold}")

    def reset_consecutive_oom_failures(self):
        """
        Reset consecutive OOM failure counter on successful container completion
        Requirement 5.2: Only count consecutive failures, reset on success
        """
        with self.lock:
            if self.consecutive_oom_failures > 0:
                logger.info(f"Consecutive OOM failures reset (was: {self.consecutive_oom_failures})")
            self.consecutive_oom_failures = 0

    def get_consecutive_oom_failures(self) -> int:
        """Get current consecutive OOM failure count"""
        with self.lock:
            return self.consecutive_oom_failures

    def should_trigger_scheduler_reset(self) -> bool:
        """
        Check if scheduler should be reset (Requirement 5.2)
        True when 3 consecutive OOM failures detected
        """
        with self.lock:
            return self.consecutive_oom_failures >= self.oom_failure_threshold
