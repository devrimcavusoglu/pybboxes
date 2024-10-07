import json

from deepdiff import DeepDiff


def assert_almost_equal(actual, desired, decimal=3, exclude_paths=None, **kwargs):
    # significant digits default value changed to 3 (from 5) due to variety in
    # results for different hardware architectures.
    diff = DeepDiff(actual, desired, significant_digits=decimal, exclude_paths=exclude_paths, **kwargs)
    assert diff == {}, f"Actual and Desired Dicts are not Almost Equal:\n {json.dumps(diff, indent=2, default=str)}"


def load_json(path: str):
    with open(path, "r") as jf:
        content = json.load(jf)
    return content
