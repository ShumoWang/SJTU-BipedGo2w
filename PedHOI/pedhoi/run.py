"""One-command, restartable, provenance-tracked reference workflow."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys,os,time,shutil,xml.etree.ElementTree as ET
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--config',default=str(ROOT/'configs/omomo_largebox_go2w.json'));parser.add_argument('--resume',action='store_true');parser.add_argument('--skip-render',action='store_true');parser.add_argument('--require-dynamics',action='store_true');args=parser.parse_args();cfgpath=Path(args.config).resolve();cfg=json.loads(cfgpath.read_text());out=(ROOT/cfg['output']).resolve();out.mkdir(parents=True,exist_ok=True)
 # Bootstrap/contact labels have only been validated for this sequence/model.
 if cfg['sequence']!='sub16_largebox_010' or cfg['source_fps']!=30 or cfg['output_fps']!=120 or cfg['time_scale']!=4 or cfg['box_dimensions_m']!=[.24,.32,.26] or cfg['box_mass_kg']!=.5:raise ValueError('This workflow is calibrated to the documented largebox sequence and settings; new data requires contact-window/robot-profile validation, not silent reuse.')
 for name in ['raw_omomo','robot_xml','omni_reference','omni_mesh','source_mesh','human_python']:
  if not Path(cfg[name]).is_file():raise FileNotFoundError(f'{name}: {cfg[name]}')
 expected=ROOT/'experiments/resmimic_rolling_v2/build_reference.py';sys.path.insert(0,str(expected.parent));import build_reference
 if Path(cfg['robot_xml']).resolve()!=Path(build_reference.MODEL).resolve():raise ValueError('Bootstrap helpers currently require the documented Go2W XML')
 codepaths=list((ROOT/'pedhoi').glob('*.py'))+[ROOT/'experiments/human_biwheel/retarget.py',ROOT/'experiments/human_ground_clamp/build.py',ROOT/'experiments/human_ground_clamp/physics_check.py',expected]
 inputs={name:dict(path=cfg[name],sha256=sha(cfg[name])) for name in ['raw_omomo','robot_xml','omni_reference','omni_mesh','source_mesh']}
 xml=ET.parse(cfg['robot_xml']).getroot();meshdir=Path(cfg['robot_xml']).parent/xml.find('compiler').get('meshdir','')
 for mesh in xml.findall('.//asset/mesh'):
  path=(meshdir/mesh.get('file')).resolve();inputs['robot_mesh:'+mesh.get('file')]=dict(path=str(path),sha256=sha(path))
 for path in sorted((Path(cfg['smplx_models'])/'smplx').glob('SMPLX_*.npz')):inputs['body_model:'+path.name]=dict(path=str(path),sha256=sha(path))
 runtime={name:version(name) for name in ['numpy','scipy','mujoco','trimesh','imageio','Pillow','matplotlib']};runtime['python']=sys.version
 codes={str(p.relative_to(ROOT)):sha(p) for p in codepaths};fingerprint=hashlib.sha256(json.dumps(dict(config=cfg,inputs=inputs,code=codes,runtime=runtime),sort_keys=True).encode()).hexdigest();manifestpath=out/'manifest.json';previous=json.loads(manifestpath.read_text()) if manifestpath.exists() else {};reuse=args.resume and previous.get('fingerprint')==fingerprint
 manifest=dict(sequence=cfg['sequence'],config=cfg,inputs=inputs,code=codes,runtime_versions=runtime,fingerprint=fingerprint,stages=previous.get('stages',{}) if reuse else {},status='RUNNING');manifestpath.write_text(json.dumps(manifest,indent=2));(out/'config.snapshot.json').write_text(json.dumps(cfg,indent=2))
 logs=out/'logs';logs.mkdir(exist_ok=True);env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='egl');os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
 dirty=False
 def stage(name,outputs,func):
  nonlocal dirty
  if reuse and not dirty and manifest['stages'].get(name,{}).get('status')=='complete' and all(Path(p).exists() and sha(p)==manifest['stages'][name].get('outputs',{}).get(str(Path(p).relative_to(out))) for p in outputs):print('Reuse',name,flush=True);return
  dirty=True
  print('Run',name,flush=True);start=time.time()
  try:func();manifest['stages'][name]=dict(status='complete',elapsed_s=time.time()-start,outputs={str(Path(p).relative_to(out)):sha(p) for p in outputs})
  except Exception as exc:manifest['stages'][name]=dict(status='failed',error=str(exc));manifest['status']='FAILED';manifestpath.write_text(json.dumps(manifest,indent=2));raise
  manifestpath.write_text(json.dumps(manifest,indent=2))
 def command(name,argv,extra=None):
  with (logs/(name+'.log')).open('w') as log:subprocess.run(argv,cwd=ROOT,env=env|dict(extra or {}),stdout=log,stderr=subprocess.STDOUT,check=True)
 human=out/'human';rolling=out/'rolling_seed';ground=out/'ground_seed'
 for p in [human,rolling,ground]:p.mkdir(exist_ok=True)
 stage('01_human',[human/'human_source.npz',human/'source.json'],lambda:command('01_human',[cfg['human_python'],'-m','pedhoi.extract',str(cfgpath),str(human)]))
 def rolling_stage():
  for f in ['human_source.npz','source.json']:shutil.copy2(human/f,rolling/f)
  command('02_rolling',[sys.executable,str(ROOT/'experiments/human_biwheel/retarget.py')],{'PEDHOI_STAGE_DIR':str(rolling)})
 stage('02_rolling',[rolling/'reference.npz'],rolling_stage)
 stage('03_ground_clamp',[ground/'reference.npz'],lambda:command('03_ground_clamp',[sys.executable,str(ROOT/'experiments/human_ground_clamp/build.py')],{'PEDHOI_STAGE_DIR':str(ground),'PEDHOI_SEED_DIR':str(rolling)}))
 from .object_prior import run as prior
 from .retarget_object import run as retarget
 from .validate_export import run as validate
 stage('04_interaction',[out/'interaction.npz',out/'object_prior_report.json'],lambda:prior(cfg,out,human/'human_source.npz',ground/'reference.npz'))
 stage('05_object_retarget',[out/'optimization.npz',out/'retarget_report.json',out/'scene.xml',out/'reference.npz'],lambda:retarget(cfg,out,ground/'reference.npz'))
 # Validation is intentionally refreshed: no stale acceptance from another trajectory.
 acceptance=validate(cfg,out)
 if acceptance['kinematic_reference_ready']:
  stage('06_free_payload_test',[out/'physics_report.json',out/'free_box_rollout.npz'],lambda:command('06_free_payload_test',[sys.executable,str(ROOT/'experiments/human_ground_clamp/physics_check.py')],{'PEDHOI_STAGE_DIR':str(out),'PEDHOI_CONTACT_FRAME':'object'}))
 acceptance=validate(cfg,out)
 if not args.skip_render:
  from .render import run as render
  stage('07_comparison',[out/'comparison.mp4',out/'validation_summary.png'],lambda:render(cfg,out,human/'human_source.npz',ground/'reference.npz'))
 manifest['stages']['05_object_retarget']['outputs']['reference.npz']=sha(out/'reference.npz')
 manifest.update(status=acceptance['status'],reference_sha256=acceptance['reference_sha256'],kinematic_reference_ready=acceptance['kinematic_reference_ready'],training_ready=False,payload_manipulation_validated=False);manifestpath.write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(status=manifest['status'],output=str(out),training_ready=False),indent=2));return 0 if acceptance['kinematic_reference_ready'] and not args.require_dynamics else 3
if __name__=='__main__':raise SystemExit(main())
