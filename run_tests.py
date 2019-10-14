# Helper script for running automated unittests

import unittest
print("Running all tests...")
loader = unittest.TestLoader()
start_dir = 'api/test'
suite = loader.discover(start_dir)

runner = unittest.TextTestRunner()
runner.run(suite)
