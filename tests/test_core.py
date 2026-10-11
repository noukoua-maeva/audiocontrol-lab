"""Offline scientific correctness and data-integrity tests; no model downloads."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from audiocontrol.evaluate import cluster_interval, evaluate, pair_results
from audiocontrol.experiment import generate, load_config
from audiocontrol.metrics import measure, mono_float

ROOT = Path(__file__).resolve().parents[1]


class AcousticTests(unittest.TestCase):
    def test_amplitude_doubling_is_six_db(self):
        t = np.arange(16000)/16000
        x = 0.1*np.sin(2*np.pi*440*t)
        delta = measure(2*x,16000)['rms_dbfs']-measure(x,16000)['rms_dbfs']
        self.assertAlmostEqual(delta, 20*np.log10(2), places=8)

    def test_fundamental_on_known_tones(self):
        for sr in (8000,16000,44100):
            for hz in (220,440,660):
                t = np.arange(sr)/sr
                self.assertAlmostEqual(measure(0.2*np.sin(2*np.pi*hz*t),sr)['f0_hz'],hz,delta=5)

    def test_pulse_count_and_rate(self):
        sr=16000
        t=np.arange(3*sr)/sr
        x=sum(0.3*np.exp(-0.5*((t-s)/0.018)**2)*np.sin(2*np.pi*440*t) for s in (0.5,1.5,2.5))
        self.assertEqual(measure(x,sr)['event_count'],3)
        self.assertEqual(measure(x,sr)['event_rate_hz'],1)

    def test_silence_has_no_pitch(self):
        result=measure(np.zeros(16000),16000)
        self.assertTrue(result['silent'])
        self.assertIsNone(result['f0_hz'])
        self.assertEqual(result['event_count'],0)

    def test_pcm_conversion_before_stereo_averaging(self):
        pcm=np.array([[32767,32767],[-32768,-32768]],dtype=np.int16)
        self.assertTrue(np.allclose(mono_float(pcm),[32767/32768,-1]))
        self.assertTrue(np.allclose(mono_float(np.array([0,128,255],dtype=np.uint8)),[-1,0,127/128]))

    def test_nonfinite_and_short_audio_rejected(self):
        for x in (np.array([np.nan]*16000),np.ones(10)):
            with self.assertRaises(ValueError): measure(x,16000)

    def test_clipping_is_reported_without_normalising(self):
        result=measure(np.ones(16000)*1.2,16000)
        self.assertEqual(result['clipping_fraction'],1)
        self.assertGreater(result['rms_dbfs'],0)


class ExperimentTests(unittest.TestCase):
    def test_unchanged_reversed_and_missing_pairs_fail(self):
        base=dict(pair_id='test',guidance_scale=3.5,seed=17,attribute='pitch',silent=False,clipping_fraction=0)
        for high in (220,110,None):
            rows=[dict(base,level='low',f0_hz=220),dict(base,level='high',f0_hz=high)]
            self.assertFalse(pair_results(rows)[0]['success'])

    def test_silent_pairs_fail_even_with_spurious_measurements(self):
        base=dict(pair_id='test',guidance_scale=3.5,seed=17,attribute='pitch',clipping_fraction=0)
        rows=[dict(base,level='low',f0_hz=220,silent=True),dict(base,level='high',f0_hz=660,silent=False)]
        result=pair_results(rows)[0]
        self.assertFalse(result['success'])
        self.assertIsNone(result['delta'])

    def test_resume_finalizes_after_interrupted_status_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=ROOT/'configs/calibration.json'
            manifest=generate(cfg,tmp)
            manifest['status']='incomplete'
            (Path(tmp)/'manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(generate(cfg,tmp)['status'],'complete')

    def test_end_to_end_resume_and_configuration_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=ROOT/'configs/calibration.json'
            first=generate(cfg,tmp)
            self.assertEqual(first['status'],'complete')
            summaries=evaluate(tmp)
            self.assertEqual(len(summaries),3)
            self.assertTrue(all(r['success_rate']==1 for r in summaries))
            self.assertTrue(all(r['delta_ci_low'] is None for r in summaries))
            self.assertEqual(generate(cfg,tmp)['records'],first['records'])
            report=(Path(tmp)/'report.html').read_text()
            self.assertIn('NOT model outputs',report)
            self.assertEqual(report.count('<audio '),6)
            changed=load_config(cfg)
            changed['seeds']=[99]
            newcfg=Path(tmp)/'changed.json'
            newcfg.write_text(json.dumps(changed))
            with self.assertRaises(ValueError): generate(newcfg,tmp)

    def test_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=generate(ROOT/'configs/calibration.json',tmp)
            path=Path(tmp)/manifest['records'][0]['file']
            sr,audio=wavfile.read(path)
            wavfile.write(path,sr,audio*0)
            with self.assertRaises(ValueError): evaluate(tmp)
            with self.assertRaises(ValueError): generate(ROOT/'configs/calibration.json',tmp)

    def test_incomplete_run_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=generate(ROOT/'configs/calibration.json',tmp)
            manifest['records'].pop()
            (Path(tmp)/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): evaluate(tmp)

    def test_bootstrap_uses_equal_prompt_weights(self):
        rows=[{'pair_id':'a','delta':0}]*10+[{'pair_id':'b','delta':10}]
        point,lo,hi,count=cluster_interval(rows,'delta')
        self.assertEqual(point,5)
        self.assertEqual(count,2)
        self.assertLessEqual(lo,point)
        self.assertGreaterEqual(hi,point)
        self.assertIsNone(cluster_interval(rows[:1],'delta')[1])

    def test_unsafe_or_duplicate_design_rejected(self):
        for mutation in ('unsafe','duplicate','nan'):
            cfg=load_config(ROOT/'configs/calibration.json')
            if mutation=='unsafe': cfg['pairs'][0]['id']='../escape'
            if mutation=='duplicate': cfg['seeds']=[1,1]
            if mutation=='nan': cfg['guidance_scales']=[float('nan')]
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'bad.json'
                path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError): load_config(path)


if __name__=='__main__':
    unittest.main()
