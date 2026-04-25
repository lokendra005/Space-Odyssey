import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

from .scoring_engine import apply_proposal, calculate_crew_survival_index
from .specialist_bots import Engineer, Pilot, Commander
from .drift_schedule import DRIFT_EVENTS
from .reward import compute_reward

class ProcurementDriftEnv(gym.Env):
    """
    Procurement Drift Environment (formerly StationEnv).
    An Overseer agent must APPROVE or VETO specialist proposals while 
    managing critical resources and scheduled space hazards.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Action Space: 0 = VETO, 1 = APPROVE
        self.action_space = spaces.Discrete(2)

        # Observation space dictionary: current state values and proposal text
        self.observation_space = spaces.Dict({
            "state": spaces.Dict({
                "oxygen": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "power": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "fuel": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "hull_integrity": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "crew_morale": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
                "step_count": spaces.Discrete(31)
            }),
            "proposal_description": spaces.Text(min_length=0, max_length=1000)
        })

        # Initialize specialists
        self.bots = [Engineer(), Pilot(), Commander()]
        self.state = None
        self.current_proposal = None
        self.step_limit = 30

    def _get_obs(self) -> Dict[str, Any]:
        """Returns the current observation in the defined space format."""
        obs_state = {
            k: (np.array([float(v)], dtype=np.float32) if k != "step_count" else v)
            for k, v in self.state.items()
        }
        return {
            "state": obs_state,
            "proposal_description": self.current_proposal["description"] if self.current_proposal else ""
        }

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Resets the environment to initial state."""
        super().reset(seed=seed)
        
        # Initial state setup
        self.state = {
            "oxygen": 100.0,
            "power": 100.0,
            "fuel": 100.0,
            "hull_integrity": 100.0,
            "crew_morale": 100.0,
            "step_count": 0
        }
        
        # Apply initial drift if any (Step 0)
        self._apply_drift()
        
        # Generate the first proposal for the agent
        self._generate_new_proposal()
        
        return self._get_obs(), {}

    def _apply_drift(self):
        """Applies silent state updates from the drift schedule."""
        step = self.state["step_count"]
        if step in DRIFT_EVENTS:
            effects = DRIFT_EVENTS[step]["effects"]
            for key, val in effects.items():
                if key in self.state:
                    self.state[key] = max(0.0, self.state[key] + val)

    def _generate_new_proposal(self):
        """Selects a bot and generates a new specialist proposal."""
        # Cycle through specialists
        bot_idx = self.state["step_count"] % len(self.bots)
        bot = self.bots[bot_idx]
        self.current_proposal = bot.generate_proposal(self.state)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        Executes one step in the environment.
        1. Applies action (Approve/Veto) to the proposal seen in the observation.
        2. Increments step.
        3. Applies drift for the new step.
        4. Generates next proposal.
        """
        
        # Apply current proposal if APPROVED (1), discard if VETOED (0)
        if action == 1 and self.current_proposal:
            self.state = apply_proposal(self.state, self.current_proposal)
        
        # Increment step count
        self.state["step_count"] += 1
        
        # Apply drift for the current step
        self._apply_drift()
        
        # Check termination: any life-support resource reaches 0
        terminated = (
            self.state["oxygen"] <= 0 or 
            self.state["power"] <= 0 or 
            self.state["hull_integrity"] <= 0
        )
        
        # Episode length limit
        truncated = self.state["step_count"] >= self.step_limit
        
        # Generate proposal for the next step observation (if not finished)
        if not (terminated or truncated):
            self._generate_new_proposal()
        
        # Compute reward calling the external reward logic
        reward = compute_reward(self.state, terminated)
        
        # Prep info dictionary
        info = {"survival_index": calculate_crew_survival_index(self.state)}
        
        return self._get_obs(), reward, terminated, truncated, info
