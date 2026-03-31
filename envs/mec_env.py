import torch
import numpy as np

from gym import spaces
from gym.utils import seeding
from gym.spaces import prng

class Env():
    def __init__(self, alpha, beta, B, N0, pi, K, ser, fi_m, fi_l, Ci=None, Di=None, hi=None):

        self.alpha, self.beta = alpha, beta
        self.B, self.N0, self.pi, self.K, self.ser = B, N0, pi, K, ser
        if Di is not None and Ci is not None and hi is not None:
            self.Di = Di
            self.Ci = Ci
            self.hi = hi
            self.is_fixed = True
        else:
            self.Di = np.random.uniform(300, 500, self.K)
            self.Ci = np.random.uniform(900, 1100, self.K)
            self.hi = pow(np.random.uniform(50, 2000, (self.K, self.ser)), -3)
            self.is_fixed = False
        self.fi_m, self.fi_l = fi_m, fi_l

        self.reward = np.zeros(self.K)
        self.np_random, _ = seeding.np_random()
        self.discrete_action_space = spaces.Discrete(self.ser + 1)
        self.continuous_action_space_0 = spaces.Box(low=1.0, high=1.0, shape=(1,), dtype=np.float32)

        continuous_low_high = [(0.0, 1.0)] * self.ser
        self.continuous_action_space_1 = [spaces.Box(low=low, high=high, shape=(1,), dtype=np.float32) for low, high in continuous_low_high]
        self.action_space = spaces.Tuple((self.discrete_action_space, self.continuous_action_space_0,)
                                         + tuple(self.continuous_action_space_1))

        self.single_observation_space = spaces.Box(
            low=0.0,
            high=1100,
            shape=(2 + self.ser + 1 + self.ser,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1100,
            shape=(self.K * (2 + self.ser + 1 + self.ser),),
            dtype=np.float32
        )

    def seed(self, seed=None):

        prng.seed(seed)
        return [seed]

    def to_numpy_if_tensor(self, x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    def step(self, act, act_param):
        """
            Placeholder for a proprietary module that is temporarily omitted
            due to confidentiality agreements with industry partners.

            The full implementation will be released upon acceptance of the paper.
            """
        raise NotImplementedError(
            "This function is temporarily unavailable in the public version "
            "due to confidentiality agreements with industry partners."
        )

    def reset(self):
        if not self.is_fixed:
            self.Di = np.random.uniform(300, 500, self.K)
            self.Ci = np.random.uniform(900, 1100, self.K)
            self.hi = pow(np.random.uniform(50, 2000, (self.K, self.ser)), -3)

        state, reward, done, _ = self.step(np.random.randint(0, self.ser+1, size=(self.K,)), np.random.uniform(0, 1, (self.K, self.action_space.spaces[0].n)))
        return state