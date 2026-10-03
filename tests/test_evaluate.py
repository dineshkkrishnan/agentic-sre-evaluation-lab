import copy
import json
import unittest
from pathlib import Path
from evaluate import baseline,public_case,score,validate_cases
from ollama_adapter import payload
CASES=json.loads(Path('data/incidents.json').read_text())
class EvaluationTests(unittest.TestCase):
    def perfect(self):return [dict(case_id=c['case_id'],diagnosis=c['expected_diagnosis'],action=c['allowed_actions'][0],evidence_ids=c['required_evidence']) for c in CASES]
    def test_perfect_contract(self):self.assertEqual(score(CASES,self.perfect())['overall_pass_rate'],1)
    def test_missing_prediction_denominator(self):
        r=score(CASES,self.perfect()[:-1]);self.assertEqual(r['prediction_coverage'],7/8);self.assertEqual(r['overall_pass_rate'],7/8)
    def test_unsafe_action(self):
        p=self.perfect();p[0]['action']='delete_data';self.assertEqual(score(CASES,p)['disallowed_or_invalid_action_rate'],1/8)
    def test_fabricated_citation(self):
        p=self.perfect();p[0]['evidence_ids']=['fake'];self.assertFalse(score(CASES,p)['details'][0]['evidence_contract_met'])
    def test_required_evidence(self):
        p=self.perfect();p[0]['evidence_ids']=[];self.assertFalse(score(CASES,p)['details'][0]['passed'])
    def test_invalid_schema(self):
        p=self.perfect();p[0]['diagnosis']=[];self.assertEqual(score(CASES,p)['schema_valid_rate'],7/8)
    def test_duplicate_and_unknown(self):
        for p in [self.perfect()+[self.perfect()[0]],[{'case_id':'unknown'}]]:
            with self.assertRaises(ValueError):score(CASES,p)
    def test_no_labels_in_model_prompt(self):
        text=json.dumps(payload(CASES[0],'test'))
        self.assertNotIn('expected_diagnosis',text);self.assertNotIn('allowed_actions',text);self.assertNotIn('required_evidence',text)
    def test_injection_ignored_by_baseline(self):self.assertEqual(baseline(public_case(CASES[4]))['action'],'inspect_dependency')
    def test_zero_predictions(self):self.assertEqual(score(CASES,[])['overall_pass_rate'],0)
    def test_unknown_cost_not_zero(self):self.assertIsNone(score(CASES,self.perfect())['cost_usd'])
    def test_invalid_latency(self):
        p=self.perfect();p[0]['latency_ms']=float('nan')
        with self.assertRaises(ValueError):score(CASES,p)
    def test_latency_coverage(self):
        p=self.perfect();p[0]['latency_ms']=4
        r=score(CASES,p);self.assertEqual(r['median_reported_latency_ms'],4);self.assertEqual(r['latency_coverage'],1/8)
    def test_dataset_duplicates(self):
        with self.assertRaises(ValueError):validate_cases(CASES+[CASES[0]])
if __name__=='__main__':unittest.main()
