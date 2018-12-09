import argparse
import os

from lab.experiment import Experiment
from lab_globals import lab

from main.main import main

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
args = parser.parse_args()

game = args.game
state = args.state

experiment = Experiment(lab=lab, name="{}:{}".format(game, state), run_file=__file__ ,
                    comment="experimenting with {}".format(game), check_repo_dirty=False,
                    project_root=os.path.join(os.path.dirname(__file__), '../'))

main(experiment)
