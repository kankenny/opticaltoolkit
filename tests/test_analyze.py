import pytest
from sklearn.datasets import load_digits

from optical_toolkit.analyze.analyze import analyze_image_dataset


@pytest.mark.slow
def test_analyze():
    digits = load_digits()
    X = digits.images
    y = digits.target

    analyze_image_dataset(X, y, output_path="examples/analyze/analysis.pdf")
