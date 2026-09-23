"""Representation and end-to-end artifact checks; physics failure stays visible."""
from pathlib import Path
import json,unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from pedhoi.object_prior import corners,laplacian
from pedhoi.geometry import object_outward_normal,pair_distance
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/omni_ground_clamp'
class RepresentationTests(unittest.TestCase):
 def test_rotated_ground_support(self):
  half=np.array([.12,.16,.13]);rot=R.from_euler('xyz',[25,18,43],degrees=True);support=np.abs(rot.as_matrix()[2])@half
  self.assertAlmostEqual(np.min(rot.apply(corners(half))[:,2])+support,0.,places=12)
  self.assertGreater(abs(support-half[2]),.02) # fixed half-height would penetrate
 def test_interaction_graph_invariance(self):
  p=np.vstack([corners([.2,.3,.4]),[.6,.4,.8],[-.4,.7,.9]]);L=laplacian(p);rot=R.from_euler('xyz',[23,14,71],degrees=True)
  np.testing.assert_allclose(L@(p+[3,4,5]),L@p,atol=1e-14)
  np.testing.assert_allclose(L@rot.apply(p),rot.apply(L@p),atol=1e-14)
 def test_zero_distance_with_separate_witness_is_not_contact(self):
  def ambiguous_query(model,data,first,second,limit,segment):
   segment[:]=[0,0,0,0,.01,0]
   return 0.
  with patch('mujoco.mj_geomDistance',side_effect=ambiguous_query):
   self.assertAlmostEqual(pair_distance(None,SimpleNamespace(contact=[]),1,2),.01)
 def test_wxyz_conversion(self):
  q=R.from_euler('y',80,degrees=True).as_quat();disk=q[[3,0,1,2]]
  np.testing.assert_allclose(R.from_quat(disk[[1,2,3,0]]).apply([0,0,1]),[np.sin(np.radians(80)),0,np.cos(np.radians(80))])
class ArtifactTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not (OUT/'acceptance.json').exists():raise unittest.SkipTest('Run configured workflow first')
 def test_omni_coordinate_audit(self):
  r=json.loads((OUT/'object_prior_report.json').read_text());self.assertLess(r['omni_object_rotation_alignment_max_deg'],.001)
 def test_independent_geometry_and_semantic_gates(self):
  a=json.loads((OUT/'acceptance.json').read_text());self.assertTrue(a['kinematic_reference_ready'],a);self.assertTrue(all(a['gates'].values()));self.assertFalse(a['training_ready']);self.assertFalse(a['payload_manipulation_validated'])
 def test_contact_normals_are_opposing_box_sides(self):
  z=np.load(OUT/'reference.npz');m=mujoco.MjModel.from_xml_path(str(OUT/'scene.xml'));d=mujoco.MjData(m);obj=m.geom('payload_box').id
  gids=[np.where(m.geom_bodyid==m.body(leg+'_wheel_link').id)[0][0] for leg in ['FL','FR']]
  for k in np.flatnonzero(z['grip_blend']>.999):
   d.qpos[:23]=z['qpos'][k];d.qpos[23:26]=z['object_pos'][k];d.qpos[26:]=z['object_quat'][k];mujoco.mj_forward(m,d);rot=R.from_quat(z['object_quat'][k,[1,2,3,0]])
   for i,g in enumerate(gids):
    world_normals,present=object_outward_normal(m,d,g,obj);normals=[rot.inv().apply(n) for n in world_normals]
    self.assertLess(abs(pair_distance(m,d,g,obj)),.003)
    self.assertTrue(normals,f'No contact normal or separated witness at {k}, side {i}')
    self.assertGreater(max(n[1]*(1 if i==0 else -1) for n in normals),.8,f'Contact is not on the intended side at {k}')
 def test_reference_export(self):
  z=np.load(OUT/'episode.npz');meta=json.loads(str(z['metadata_json']));self.assertEqual(z['qpos'].shape[1],23);self.assertEqual(z['qvel'].shape[1],22);self.assertEqual(len(z['joint_names']),16);self.assertEqual(meta['object_quaternion'],'wxyz');self.assertEqual(z['object_ang_vel_world'].shape,(len(z['qpos']),3));self.assertTrue(np.isfinite(z['object_ang_vel_world']).all());self.assertFalse(bool(z['training_ready']));np.testing.assert_allclose(z['robot_root_quat_wxyz'],z['qpos'][:,3:7]);np.testing.assert_allclose(z['object_quat_wxyz'],z['object_quat'])
 def test_improved_object_orientation(self):
  r=json.loads((OUT/'retarget_report.json').read_text());self.assertLess(r['rotation_error_after_carry_rms_deg'],r['rotation_error_before_carry_rms_deg']-10);self.assertTrue(r['rear_qpos_unchanged']);self.assertTrue(r['root_unchanged'])
if __name__=='__main__':unittest.main()
