from .affinities_with_glia import create_experiment_main as create_affinities_with_glia, get_parser as _get_parser_affinities_with_glia
from .templates import make_architecture, make_training, make_architecture_no_docker, make_training_no_docker

def create_experiment():
    import argparse
    parser = argparse.ArgumentParser()

    mapping = {
        'affinities-with-glia': create_affinities_with_glia
    }

    parser.add_argument('command', choices=tuple(mapping.keys()))
    command, argv = parser.parse_known_args()
    mapping[command.command](argv=argv)


