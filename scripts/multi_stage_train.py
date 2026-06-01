#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import subprocess

def write_temp_config(base_cfg, out_path):
    with open(out_path, 'w') as f:
        yaml.safe_dump(base_cfg, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yml')
    parser.add_argument('--stages', default='generated,khatt,muharaf')
    parser.add_argument('--epochs', default='')
    parser.add_argument('--continue-on-fail', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    base_checkpoint_dir = cfg.get('training', {}).get('checkpoint_dir', 'checkpoints')

    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    epochs_list = []
    if args.epochs:
        epochs_list = [int(x) for x in args.epochs.split(',')]

    resume_path = None
    for idx, stage in enumerate(stages):
        stage_cfg = dict(cfg)
        stage_cfg.setdefault('data', {})['dataset'] = stage
        stage_ckpt = os.path.join(base_checkpoint_dir, f'stage_{stage}')
        stage_cfg.setdefault('training', {})['checkpoint_dir'] = stage_ckpt

        os.makedirs(stage_ckpt, exist_ok=True)

        temp_cfg_path = os.path.join(stage_ckpt, f'config_{stage}.yml')
        write_temp_config(stage_cfg, temp_cfg_path)

        cmd = [sys.executable, 'scripts/train.py', '--config', temp_cfg_path]
        if len(epochs_list) > idx:
            cmd += ['--epochs', str(epochs_list[idx])]
        elif len(epochs_list) == 1:
            cmd += ['--epochs', str(epochs_list[0])]

        if resume_path:
            cmd += ['--resume', resume_path]

        print('Running stage:', stage)
        print('Command:', ' '.join(cmd))
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            print(f"Stage {stage} failed with return code {ret.returncode}")
            if not args.continue_on_fail:
                sys.exit(ret.returncode)
        # next stage will resume from this stage's best model if available
        candidate = os.path.join(stage_ckpt, 'best_model.pt')
        if os.path.exists(candidate):
            resume_path = candidate
        else:
            resume_path = None

    print('Multi-stage training finished')


if __name__ == '__main__':
    main()
