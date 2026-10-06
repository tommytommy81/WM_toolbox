"""
Command-line interface for the STWM MEG pipeline.

Usage
-----
python cli/stwm.py sensor              --subject S1   # individual sensor analysis
python cli/stwm.py sensor-group                       # group sensor statistics
python cli/stwm.py inspect             --subject S1   # visual inspection report
python cli/stwm.py source              --subject S1   # individual source analysis
python cli/stwm.py source-group                       # group source statistics
python cli/stwm.py connectivity        --subject S1   # individual connectivity
python cli/stwm.py connectivity-group  --step template|pairs|stats|viz

All commands accept --config PATH (default: config.yaml at the repository root).
"""

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import CONFIG_PATH, load_config  # noqa: E402


def _sensor(config, args):
    from utils.sensor import run_individual_analysis
    run_individual_analysis(config)


def _sensor_group(config, args):
    from utils.sensor import run_group_statistics
    run_group_statistics(config)


def _inspect(config, args):
    from utils.sensor import VisualInspector
    VisualInspector(config).create_comparison_report()


def _source(config, args):
    from utils.source import run_source_analysis
    run_source_analysis(config)


def _source_group(config, args):
    from utils.source import run_source_group_statistics
    run_source_group_statistics(config)


def _connectivity(config, args):
    from utils.connectivity import run_connectivity_analysis
    run_connectivity_analysis(config)


def _connectivity_group(config, args):
    from utils import connectivity as c
    {'template': c.src_average,
     'pairs': c.pairs_identification,
     'stats': c.connectivity_statistics,
     'viz': c.connectivity_statistics_visualization}[args.step](config)


COMMANDS = {
    'sensor':             (_sensor, True, 'Individual sensor-space analysis'),
    'sensor-group':       (_sensor_group, False, 'Group sensor-space statistics'),
    'inspect':            (_inspect, True, 'Visual inspection report for one subject'),
    'source':             (_source, True, 'Individual source-space analysis'),
    'source-group':       (_source_group, False, 'Group source-space statistics'),
    'connectivity':       (_connectivity, True, 'Individual connectivity analysis'),
    'connectivity-group': (_connectivity_group, False, 'Group connectivity step'),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description='STWM MEG analysis pipeline')
    sub = parser.add_subparsers(dest='command', required=True)
    for name, (_, per_subject, help_text) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument('--config', default=CONFIG_PATH, help='Path to configuration file')
        if per_subject:
            p.add_argument('--subject', help='Subject name (overrides config file)')
        if name == 'connectivity-group':
            p.add_argument('--step', required=True, choices=['template', 'pairs', 'stats', 'viz'])

    args = parser.parse_args(argv)
    config = load_config(args.config)
    if getattr(args, 'subject', None):
        config['subject']['subject_name'] = args.subject

    try:
        COMMANDS[args.command][0](config, args)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
