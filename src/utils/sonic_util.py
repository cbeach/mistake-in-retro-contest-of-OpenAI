import traceback
import gym
import numpy as np
import os
from os import environ, path
import sys
from itertools import combinations
import retro_contest


import retro
from .atari_wrappers import FrameStack, WarpFrame, GameOverAwareWrapper
from gym import spaces
#import gym_remote.client as grc

'''
the comments below is all games and states in Sonic series.
Before training the model, you should get the *.md files for Sonic game and import them by openAI.retro(https://github.com/openai/retro), for example: python -m retro.import /path/to/your/ROMs/directory/
'''

'''game = ['SonicTheHedgehog-Genesis', 'SonicTheHedgehog2-Genesis', 'SonicAndKnuckles3-Genesis']
state1 = ['SpringYardZone.Act3', 'SpringYardZone.Act2', 'GreenHillZone.Act3', 'GreenHillZone.Act1',
          'StarLightZone.Act2', 'StarLightZone.Act1', 'MarbleZone.Act2', 'MarbleZone.Act1', 'MarbleZone.Act3',
          'ScrapBrainZone.Act2', 'LabyrinthZone.Act2', 'LabyrinthZone.Act1', 'LabyrinthZone.Act3']
state2 = ['EmeraldHillZone.Act1', 'EmeraldHillZone.Act2', 'ChemicalPlantZone.Act2', 'ChemicalPlantZone.Act1',
          'MetropolisZone.Act1', 'MetropolisZone.Act2', 'OilOceanZone.Act1', 'OilOceanZone.Act2',
          'MysticCaveZone.Act2', 'MysticCaveZone.Act1', 'HillTopZone.Act1', 'CasinoNightZone.Act1',
          'WingFortressZone', 'AquaticRuinZone.Act2', 'AquaticRuinZone.Act1']
state3 = ['LavaReefZone.Act2', 'CarnivalNightZone.Act2', 'CarnivalNightZone.Act1', 'MarbleGardenZone.Act1',
          'MarbleGardenZone.Act2', 'MushroomHillZone.Act2', 'MushroomHillZone.Act1', 'DeathEggZone.Act1',
          'DeathEggZone.Act2', 'FlyingBatteryZone.Act1', 'SandopolisZone.Act1', 'SandopolisZone.Act2',
          'HiddenPalaceZone', 'HydrocityZone.Act2', 'IcecapZone.Act1', 'IcecapZone.Act2', 'AngelIslandZone.Act1',
          'LaunchBaseZone.Act2', 'LaunchBaseZone.Act1']'''

game_wrappers = {
    'SuperMarioBros': GameOverAwareWrapper,
}


def get_models_dir(game, state):
    base_data_dir = path.join(environ.get('DATA_DIR', environ.get('HOME', '.')), 'game_playing')
    return path.join(base_data_dir, 'models/{}/'.format(game, state))


def get_replay_dir(game, state):
    base_data_dir = path.join(environ.get('DATA_DIR', environ.get('HOME', '.')), 'game_playing')
    return path.join(base_data_dir, 'replays/{}/{}/'.format(game, state))


def list_envs():
    return {game: retro.data.list_states(game)
            for game in retro.data.list_games()}

def retro_make(game, state=retro.State.DEFAULT, discrete_actions=False, bk2dir=None, record='.'):
    use_restricted_actions = retro.Actions.FILTERED
    if discrete_actions:
        use_restricted_actions = retro.Actions.DISCRETE
    try:
        env = retro.make(game, state, scenario='deep_thought', record=record, use_restricted_actions=use_restricted_actions)
    except Exception as e:
        print('EXCEPTION in retro_make')
        print(traceback.format_exc())
        env = retro.make(game, state, use_restricted_actions=use_restricted_actions)
    if bk2dir:
        env.auto_record(bk2dir)
    env = retro_contest.StochasticFrameSkip(env, n=4, stickprob=0.25)
    if game in game_wrappers:
        env = game_wrappers[game](env)
    else:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=4500)

    return env


def make_env(stack=True,
        scale_rew=True,
        game='SonicTheHedgehog-Genesis',
        state='LabyrinthZone.Act2'):
    """
    Create an environment with some standard wrappers.
    """
    #env = grc.RemoteEnv('tmp/sock') #contest env
    platform = game.split('-')[1].lower()

    replay_dir = get_replay_dir(game, state)
    if not path.isdir(replay_dir):
        os.makedirs(replay_dir)

    env = retro_make(game=game, state=state, record=replay_dir)
    if game.startswith("SuperMario"):
        print("a mario game")
        env = MarioDiscretizer(env, platform)
        env = GameOverAwareWrapper(env)
    else:
        env = PlatformDiscretizer(env, platform)
    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env)
    if stack:
        env = FrameStack(env, 4)
    return env

class MarioDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, platform):
        super(MarioDiscretizer, self).__init__(env)

        buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
        actions = [('B', 'A', 'DOWN', 'LEFT'),
                   ('B', 'A', 'DOWN', 'RIGHT'),
                   ('B', 'A', 'UP'),
                   ('B', 'A', 'DOWN'),
                   ('B', 'A', 'LEFT'),
                   ('B', 'A', 'RIGHT'),
                   ('B', 'DOWN', 'LEFT'),
                   ('B', 'DOWN', 'RIGHT'),
                   ('A', 'DOWN', 'LEFT'),
                   ('A', 'DOWN', 'RIGHT'),
                   ('B', 'A'),
                   ('B', 'UP'),
                   ('B', 'DOWN'),
                   ('B', 'LEFT'),
                   ('B', 'RIGHT'),
                   ('A', 'UP'),
                   ('A', 'DOWN'),
                   ('A', 'LEFT'),
                   ('A', 'RIGHT'),
                   ('B',),
                   ('A',),
                   ('UP',),
                   ('DOWN',),
                   ('LEFT',),
                   ('RIGHT',)]

        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
class PlatformDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env, platform):
        super(PlatformDiscretizer, self).__init__(env)

        if platform == 'genesis':
            buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
            actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                       ['DOWN', 'B'], ['B']]
        elif platform == 'nes':
            buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
            actions = combinations(["B", "A", "UP", "DOWN", "LEFT", "RIGHT"])
        elif platform == 'snes':
            buttons = ["B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]
            actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                       ['DOWN', 'B'], ['B']]

        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()
class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):
        return reward * 0.01

class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    '''
    if you want play a random game or state, you can
    modify reset function
    '''
    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

