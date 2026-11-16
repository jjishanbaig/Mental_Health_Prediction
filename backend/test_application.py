import json
from pathlib import Path
import csv
from io import StringIO
import subprocess
import os
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
        cls.server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '18765'], cwd=ROOT.parent, env={**os.environ, 'GEMINI_API_KEY':''}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(600):
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

    def test_rag_page_and_configuration(self):
        status,page,_=self.call('/rag')
        self.assertEqual(status,200)
        self.assertIn(b'Mental Wellness AI Assistant',page)
        self.assertIn(b'<body class="">',page)
        embedded_status,embedded_page,_=self.call('/rag?embedded=1')
        self.assertEqual(embedded_status,200)
        self.assertIn(b'<body class="embedded">',embedded_page)
        self.assertEqual(self.call('/static/rag.js')[0],200)
        status,configuration,_=self.call('/api/rag/status')
        self.assertEqual(status,200)
        self.assertIn('configured',configuration)
        if not configuration['configured']:
            response=self.call('/api/rag/chat',{'question':'What is stress?'})
            self.assertEqual(response[0],200)
            self.assertEqual(response[1]['mode'],'guide_search')
            self.assertEqual(response[1]['sources'][0]['entry_id'],'KB-001')

    def test_rag_empty_context_without_api_call(self):
        from unittest.mock import patch, MagicMock
        from backend.app import rag_service
        empty=MagicMock()
        empty.count.return_value=0
        with patch.object(rag_service,'ensure_shared_knowledge'), patch.object(rag_service,'get_collection',return_value=empty), patch.object(rag_service,'generate_embeddings') as embeddings:
            result=rag_service.gemini_answer_question('What is stress?')
            self.assertEqual(result['sources'],[])
            embeddings.assert_not_called()

    def test_shared_pdf_indexed_once(self):
        from unittest.mock import patch, MagicMock
        from backend.app import rag_service
        collection=MagicMock()
        collection.get.side_effect=[{'ids':[]},{'ids':['already-indexed']}]
        with patch.object(rag_service,'API_KEY','test-key'), patch.object(rag_service,'get_collection',return_value=collection), patch.object(rag_service,'generate_embeddings',side_effect=lambda texts,task:[[0.1,0.2] for _ in texts]) as embeddings, patch.object(Path,'write_bytes'):
            self.assertTrue(rag_service.ensure_shared_knowledge())
            self.assertTrue(rag_service.ensure_shared_knowledge())
            embeddings.assert_called_once()
            collection.add.assert_called_once()
            inserted=collection.add.call_args.kwargs
            self.assertEqual(len(inserted['documents']),24)
            self.assertEqual(inserted['metadatas'][0]['entry_id'],'KB-001')
            self.assertTrue(all(2<=m['page']<=7 for m in inserted['metadatas']))
            self.assertTrue(all(m['bundled'] for m in inserted['metadatas']))

    def test_shared_pdf_cannot_be_deleted(self):
        from unittest.mock import patch, MagicMock
        from backend.app import rag_service
        collection=MagicMock()
        collection.get.return_value={'ids':['shared'], 'metadatas':[{'bundled':True}]}
        with patch.object(rag_service,'get_collection',return_value=collection):
            with self.assertRaises(ValueError):
                rag_service.delete_document('shared')
            collection.delete.assert_not_called()

    def test_local_guide_handles_typo_and_unrelated_question(self):
        from backend.app import rag_service
        result=rag_service.guide_search_answer('what is strss')
        self.assertEqual(result['mode'],'guide_search')
        self.assertEqual(result['sources'][0]['entry_id'],'KB-001')
        self.assertIn('not an AI-generated answer',result['answer'])
        self.assertEqual(rag_service.guide_search_answer('Who won the football world cup?')['sources'],[])

    def test_provider_failure_falls_back_and_pauses_retries(self):
        from unittest.mock import patch
        import httpx
        from backend.app import rag_service
        with patch.object(rag_service,'API_KEY','test'), patch.object(rag_service,'_gemini_retry_at',0), patch.dict(os.environ,{'RAG_MODE':'auto'}), patch.object(rag_service,'gemini_answer_question',side_effect=httpx.ReadTimeout('timeout')) as cloud:
            first=rag_service.answer_question('What is stress?')
            second=rag_service.answer_question('How can I sleep better?')
            self.assertEqual(first['mode'],'guide_search')
            self.assertEqual(second['mode'],'guide_search')
            self.assertTrue(first['sources'])
            cloud.assert_called_once()

if __name__ == '__main__':
    unittest.main(verbosity=2)

