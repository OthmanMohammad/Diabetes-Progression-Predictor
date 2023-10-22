from src.model import DiabetesPredictor

def test_model_initialization():
    model_instance = DiabetesPredictor()
    assert model_instance.model is not None
