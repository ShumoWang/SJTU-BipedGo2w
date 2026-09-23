"""Reproduce the full source-to-candidate pipeline. Exit 2 = unloaded-reference validation rejection.
No source repository is modified. Local data/model paths are recorded in scripts.
"""
from pathlib import Path
import argparse,subprocess,os,sys,json
p=Path(__file__).resolve().parent
cli=argparse.ArgumentParser();cli.add_argument('--skip-render',action='store_true');args=cli.parse_args()
env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
gmr='/home/robotennis2025/anaconda3/envs/gmr/bin/python'
steps=[(gmr,'extract_human.py'),(sys.executable,'retarget.py'),(sys.executable,'refine_balance_coarse.py'),(sys.executable,'refine_balance.py'),(sys.executable,'dynamics_check.py'),(sys.executable,'balance_rollout.py'),(sys.executable,'validate_rollout.py'),(sys.executable,'export_and_validate.py'),(sys.executable,'test_pipeline.py')]
if not args.skip_render:steps.extend([(sys.executable,'render_results.py'),(sys.executable,'render_rollout.py')])
env['PED_TAG']='balance_tracking'
for interpreter,script in steps:
 print('RUN',script,flush=True);subprocess.run([interpreter,str(p/script)],env=env,check=True,cwd=p)
gate=json.loads((p/'acceptance.json').read_text());print(gate['status'])
sys.exit(0 if gate['tracking_reference_ready'] else 2)
