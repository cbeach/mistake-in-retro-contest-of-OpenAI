#!/usr/bin/env python

import argparse
import tensorflow as tf
import json
import sys
import os
from os import environ, path
sys.path.append('..')

from rollouts import PrioritizedReplayBuffer, NStepPlayer, BatchedPlayer
from envs import BatchedFrameStack, BatchedGymEnv
from utils import AllowBacktracking, make_env, list_envs, get_models_dir
from models import DQN, rainbow_models
from spaces import gym_space_vectorizer

import gym_remote.exceptions as gre

def handle_ep(steps, reward):
    with open('./rewards', 'w') as fp:
        rewards = json.load(fp)
        fp.seek(0)
        rewards.append({'steps': steps, 'reward': reward})

def get_newest_model(game, state):
    models_dir = get_models_dir(game, state)
    print('models_dir: {}'.format(models_dir))
    checkpoints = []
    if not os.path.exists(models_dir):
        os.mkdirs(models_dir)
    else:
        for i in os.listdir(models_dir):
            try:
                checkpoints.append(int(i.replace('{}-'.format(state), '').split('.')[0]))
            except ValueError as e:
                pass
        print('checkpoints: {}'.format(checkpoints))

    return max(checkpoints) if len(checkpoints) > 0 else None


def main():
    """Run DQN until the environment throws an exception."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='game: Use retro.data.list_games() to see a list of available games', type=str, default="SuperMarioBros-Nes")
    parser.add_argument('--state', help='game state: Use retro.data.list_states(game) to see a list of available starting states', type=str, default="Level1-1")
    parser.add_argument('--num_steps', help='The number of steps to train the model.', default=3000000, type=int)
    parser.add_argument('--resume_training', help='Resume training the most recent model', action='store_true')
    parser.add_argument('--show_gameplay', help='Display the agent playing the game in realtime', action='store_true')
    parser.add_argument('--save_screens', help='save screen shots to $DATA_DIR/screenshots/[GAME]/[STATE]/[RUN_ID]/', action='store_true')
    parser.add_argument('--generate_map', help='Generate a level map', action='store_true')
    parser.add_argument('--show_map', help='Show the level map panorama as it is generated', action='store_true')
    parser.add_argument('--show_map_matches', help='Show the keypoint matches in the level map. Implies --show_map=True', action='store_true')
    parser.add_argument('-l', help='list games and states', action='store_true')

    tensorboard_dir = path.join(environ.get('DATA_DIR', environ.get('HOME', '.')), 'tensorboard')
    writer = tf.summary.FileWriter(tensorboard_dir)

    args = parser.parse_args()
    if args.l:
        import retro
        games = retro.data.list_games()
        for game in games:
            print('{}: {}'.format(game, retro.data.list_states(game)))
        sys.exit(0)

    game = args.game
    state = args.state
    resume = args.resume_training

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
        model_number = get_newest_model(game, state)
        print('model_number: {}'.format(model_number))
        print('resume: {}'.format(resume))
        if resume and model_number is not None:
            model_name = path.join(get_models_dir(game, state), '{}-{}'.format(state, model_number))
            saver = tf.train.Saver()
            saver.restore(sess, model_name)
            print('loaded model {}'.format(model_name))
        else:
            model_number = 0
        optim, optimize = dqn.optimize(learning_rate=0.0001)
        #import pdb; pdb.set_trace()
        #print(sess.graph.get_tensor_by_name('layer_1:0'))
        #sys.exit(0)

        if resume and model_number > 0:
            print('resuming at model number {}'.format(model_number))
            sess.run(tf.variables_initializer(optim.variables()))
        else:
            print('creating a new model')
            sess.run(tf.global_variables_initializer())

        writer.add_graph(sess.graph)
        dqn.train(num_steps=args.num_steps, # Make sure an exception arrives before we stop.
                  initial_step=model_number,
                  player=player,
                  replay_buffer=PrioritizedReplayBuffer(500000, 0.5, 0.4, epsilon=0.1),
                  optimize_op=optimize,
                  train_interval=1,
                  target_interval=1024,
                  batch_size=16,
                  min_buffer_size=20000,
                  save_iters=2048,
                  game=game,
                  state=state,
                  generate_map=args.generate_map,
                  show_gameplay=args.show_gameplay,
                  show_map=args.show_map,
                  show_map_matches=args.show_map_matches)

if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
