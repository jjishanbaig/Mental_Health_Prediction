import json
from pathlib import Path
import csv
from io import StringIO
import subprocess
import sys
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '18765'], cwd=ROOT.parent, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                with urlopen('http://127.0.0.1:18765/health', timeout=1):
                    break
            except OSError:
                if cls.server.poll() is not None:
                    raise RuntimeError('Test server failed to start')
                time.sleep(.1)
        else:
            cls.server.terminate()
            raise RuntimeError('Test server startup timed out')
        cls.options = cls.call('/api/options')[1]
        cls.payload = {k:v[0] for k,v in cls.options.items()}

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=10)

    @staticmethod
    def call(path, payload=None, content_type='application/json'):
        body = payload if isinstance(payload, bytes) else None if payload is None else json.dumps(payload).encode()
        req = Request('http://127.0.0.1:18765'+path, data=body, headers={'Content-Type':content_type})
        try:
            response = urlopen(req, timeout=20)
        except HTTPError as exc:
            response = exc
        with response:
            content = response.read()
            if 'application/json' in response.headers.get('Content-Type',''):
                content = json.loads(content)
            return response.status, content, response.headers

    def test_home_and_static(self):
        self.assertEqual(self.call('/')[0],200)
        script=self.call('/static/app.js')
        self.assertEqual(script[0],200)
        self.assertNotIn(b'innerHTML',script[1])

    def test_prediction(self):
        status, result, headers = self.call('/api/predict',self.payload)
        self.assertEqual(status,200)
        self.assertIn(result['prediction'],['Yes','No'])
        self.assertAlmostEqual(sum(result['probabilities'].values()),1)
        self.assertEqual(headers['Cache-Control'],'no-store')

    def test_validation(self):
        for invalid in ['', 'INVALID_CATEGORY', '<img src=x onerror=alert(1)>', None, 123]:
            self.assertEqual(self.call('/api/predict',{**self.payload,'Occupation':invalid})[0],422)
        self.assertEqual(self.call('/api/predict',{**self.payload,'extra':'value'})[0],422)
        self.assertEqual(self.call('/api/predict',{})[0],422)

    def test_missing_imputation(self):
        import backend.app.main as main
        payload = main.PredictionInput(**{k:'Not specified' for k in self.payload})
        output=main.infer(payload)
        frame=pd.DataFrame([{k:np.nan for k in main.FEATURES}],columns=main.FEATURES)
        expected=main.model.predict_proba(frame)[0]
        np.testing.assert_allclose(list(output[3].values()),expected)
        self.assertFalse(pd.isna(main.model.named_steps['imputer'].transform(frame)).any())
        self.assertEqual(self.call('/api/predict',payload.model_dump())[0],200)

    def test_report_without_storage(self):
        before=set((ROOT/'reports').glob('*'))
        status, pdf, headers=self.call('/api/report',self.payload)
        self.assertEqual(status,200)
        self.assertTrue(pdf.startswith(b'%PDF-'))
        self.assertEqual(headers['Content-Type'],'application/pdf')
        self.assertEqual(before,set((ROOT/'reports').glob('*')))
        self.assertEqual(self.call('/reports/bf2f030a66b4')[0],404)

    def test_metrics_consistency(self):
        import backend.app.main as main
        metrics=json.loads((ROOT/'models/model_metrics.json').read_text())
        self.assertEqual(metrics,main.METRICS)
        self.assertEqual(metrics['group_overlap'],0)
        self.assertEqual(sum(map(sum,metrics['confusion_matrix'])),metrics['test_rows'])
        self.assertEqual(len(metrics['cross_validation']),5)
        self.assertEqual(self.call('/api/stats')[1]['metrics'],metrics)

    def test_dataset_endpoints(self):
        self.assertEqual(len(self.call('/api/data-preview?limit=1000')[1]['rows']),100)
        self.assertEqual(self.call('/api/data-preview?limit=abc')[0],422)
        self.assertEqual(self.call('/dataset')[0],200)

    def test_batch_matches_single(self):
        status,template,_=self.call('/api/batch-template')
        self.assertEqual(status,200)
        sample=list(csv.DictReader(StringIO(template.decode())))[0]
        single=self.call('/api/predict',sample)[1]
        status,content,headers=self.call('/api/predict-batch',template,'text/csv')
        self.assertEqual(status,200)
        result=list(csv.DictReader(StringIO(content.decode())))[0]
        self.assertEqual(result['predicted_treatment'],single['prediction'])
        self.assertAlmostEqual(float(result['predicted_probability']),single['confidence'])
        self.assertEqual(headers['X-Prediction-Count'],'1')
        self.assertEqual(headers['Cache-Control'],'no-store')

    def test_batch_validation_and_limits(self):
        template=self.call('/api/batch-template')[1]
        self.assertEqual(self.call('/api/predict-batch',template)[0],415)
        for content in [b'', b'wrong\nvalue\n', template.splitlines()[0]+b'\n', b'\xff']:
            self.assertEqual(self.call('/api/predict-batch',content,'text/csv')[0],422)
        lines=template.decode().splitlines()
        invalid=(lines[0]+'\nINVALID,'+lines[1].split(',',1)[1]+'\n').encode()
        status,error,_=self.call('/api/predict-batch',invalid,'text/csv')
        self.assertEqual(status,422)
        self.assertIn('Row 2',error['detail'])
        too_many=(lines[0]+'\n'+(lines[1]+'\n')*1001).encode()
        self.assertEqual(self.call('/api/predict-batch',too_many,'text/csv')[0],413)
        self.assertEqual(self.call('/api/predict-batch',b'x'*(1024*1024+1),'text/csv')[0],413)

    def test_batch_blanks_and_multiple_rows(self):
        keys=list(self.options)
        output=StringIO(newline='')
        writer=csv.writer(output)
        writer.writerow(keys)
        writer.writerow(['']*len(keys))
        writer.writerow([self.payload[k] for k in keys])
        status,content,_=self.call('/api/predict-batch',output.getvalue().encode('utf-8-sig'),'text/csv; charset=utf-8')
        self.assertEqual(status,200)
        results=list(csv.DictReader(StringIO(content.decode())))
        self.assertEqual(len(results),2)
        self.assertEqual(results[0]['Occupation'],'Not specified')
        single=self.call('/api/predict',{k:'Not specified' for k in keys})[1]
        self.assertAlmostEqual(float(results[0]['predicted_probability']),single['confidence'])

if __name__ == '__main__':
    unittest.main(verbosity=2)

