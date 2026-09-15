"""Run from backend: python -m unittest discover -s tests -v."""
import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient
from app.main import app


class CombinedAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()
        cls.fixtures = json.loads((Path(__file__).parent / 'prediction_fixtures.json').read_text(encoding='utf-8'))

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_metadata_and_route_isolation(self):
        self.assertEqual(self.client.get('/health').status_code, 200)
        careers = self.client.get('/api/careers/metadata').json()
        courses = self.client.get('/api/courses/metadata').json()
        self.assertIn('field_filter_values', careers)
        self.assertIn('grades', careers)
        self.assertIn('fields', courses)
        self.assertEqual(self.client.get('/api/courses/model-info').json()['models_loaded'], 5)
        self.assertNotIn('/predict', app.openapi()['paths'])

    def test_predictions_match_supplied_models(self):
        for fixture in self.fixtures:
            for section in ['careers', 'courses']:
                with self.subTest(section=section, field=fixture[section]['request']['field_filter']):
                    response = self.client.post(f'/api/{section}/predict', json=fixture[section]['request'])
                    self.assertEqual(response.status_code, 200, response.text)
                    data = response.json()
                    self.assertTrue(data['success'])
                    self.assertEqual(data['recommendations'], fixture[section]['recommendations'])

    def test_invalid_fields_scores_and_wrong_payload_shape(self):
        for section in ['careers', 'courses']:
            payload = self.fixtures[0][section]['request']
            invalid = {**payload, 'field_filter': 'not-a-real-field'}
            self.assertEqual(self.client.post(f'/api/{section}/predict', json=invalid).status_code, 422)
            invalid = json.loads(json.dumps(payload))
            if section == 'careers': invalid['grades']['grade_math'] = 101
            else: invalid['grade_math'] = -1
            self.assertEqual(self.client.post(f'/api/{section}/predict', json=invalid).status_code, 422)
        self.assertEqual(self.client.post('/api/courses/predict', json=self.fixtures[0]['careers']['request']).status_code, 422)
        self.assertEqual(self.client.post('/api/careers/predict', json=self.fixtures[0]['courses']['request']).status_code, 422)

    def test_no_hidden_career_rank_prediction_errors(self):
        service = app.state.career_service
        for fixture in self.fixtures:
            profile = fixture['careers']['request']
            frame = service.build_input(profile['field_filter'], profile['hobbies'], profile['grades'], profile['scores'])
            for target, bundle in service.model_bundle.items():
                if not isinstance(bundle, dict): continue
                for key in ['career_model', 'availability_model']:
                    model = bundle.get(key)
                    if model is not None:
                        with self.subTest(target=target, model=key):
                            self.assertGreater(len(model.predict(frame)), 0)


if __name__ == '__main__':
    unittest.main()
