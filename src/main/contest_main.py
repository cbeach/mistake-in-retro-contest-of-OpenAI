import argparse
import tensorflow as tf
import json
import sys
sys.path.append('..')

from rollouts import PrioritizedReplayBuffer, NStepPlayer, BatchedPlayer
from envs import BatchedFrameStack, BatchedGymEnv
from utils import AllowBacktracking, make_env, list_envs
from models import DQN, rainbow_models
from spaces import gym_space_vectorizer

import gym_remote.exceptions as gre


def main():
    """Run DQN until the environment throws an exception."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='game: Use retro.data.list_games() to see a list of available games', type=str, default="SuperMarioBros-Nes")
    parser.add_argument('--state', help='game state: Use retro.data.list_states(game) to see a list of available starting states', type=str, default="Level1-1")
    parser.add_argument('--num_steps', help='The number of steps to train the model', default=3000000, type=int)
    args = parser.parse_args()
    game = args.game
    state = args.state

    env = AllowBacktracking(make_env(stack=False, scale_rew=False, game=game, state=state))
    env = BatchedFrameStack(BatchedGymEnv([[env]]), num_images=4, concat=False)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    with tf.Session(config=config) as sess:
        dqn = DQN(*rainbow_models(sess,
                                  env.action_space.n,
                                  gym_space_vectorizer(env.observation_space),
                                  min_val=-200,
                                  max_val=200))
        player = NStepPlayer(BatchedPlayer(env, dqn.online_net), 4)
        optim, optimize = dqn.optimize(learning_rate=0.0001)
        sess.run(tf.global_variables_initializer())
        dqn.train(num_steps=args.num_steps, # Make sure an exception arrives before we stop.
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=1024,
                  batch_size=16,
                  min_buffer_size=20000,
                  save_iters=2048,
                  game=game,
                  state=state)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
