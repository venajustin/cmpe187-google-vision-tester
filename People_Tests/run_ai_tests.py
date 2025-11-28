#!/usr/bin/env python3
"""
AI Test Runner for People Detection Test Suite

This script:
1. Runs all AI test cases (P-xx, PT-xx)
2. Collects results from JSON output files
3. Generates a summary of detections
"""

import os
import sys
import json
import time
import subprocess


def find_venv_python():
    """Find the Python executable in the virtual environment"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Check common venv locations
    venv_paths = [
        os.path.join(project_root, '.venv', 'bin', 'python3'),
        os.path.join(project_root, '.venv', 'bin', 'python'),
        os.path.join(project_root, 'venv', 'bin', 'python3'),
        os.path.join(project_root, 'venv', 'bin', 'python'),
    ]

    for venv_python in venv_paths:
        if os.path.exists(venv_python):
            return venv_python

    # Fall back to system Python
    return sys.executable


class AITestRunner:
    def __init__(self, tests_dir, results_dir, python_executable):
        self.tests_dir = tests_dir
        self.results_dir = results_dir
        self.python_executable = python_executable
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def find_all_tests(self):
        """Find all AI test files (P-xx, PT-xx)"""
        test_files = []
        for filename in sorted(os.listdir(self.tests_dir)):
            if filename.endswith('.py') and (filename.startswith('P-') or filename.startswith('PT-')):
                test_files.append(os.path.join(self.tests_dir, filename))
        return test_files

    def run_single_test(self, test_file):
        """Run a single test and return its result"""
        test_name = os.path.basename(test_file)
        print(f"Running {test_name}...", end=' ')
        sys.stdout.flush()

        try:
            result = subprocess.run(
                [self.python_executable, test_file],
                capture_output=True,
                text=True,
                timeout=120
            )

            output = result.stdout + result.stderr

            # Extract pass/fail and counts from output
            import re
            # New format: TEST CASE PT-01: PASS (Expected: 1, Detected: 1)
            match = re.search(r'TEST CASE [\w-]+: (PASS|FAIL) \(Expected: (\d+), Detected: (\d+)\)', output)
            if match:
                status = match.group(1)
                expected = int(match.group(2))
                detected = int(match.group(3))
                symbol = '✓' if status == 'PASS' else '✗'
                print(f'{symbol} {status} (Expected: {expected}, Detected: {detected})')
                return {'status': status, 'expected': expected, 'detected': detected, 'output': output}
            elif 'ERROR' in output:
                print('ERROR')
                return {'status': 'ERROR', 'expected': 0, 'detected': 0, 'output': output}
            else:
                print('UNKNOWN')
                return {'status': 'UNKNOWN', 'expected': 0, 'detected': 0, 'output': output}

        except subprocess.TimeoutExpired:
            print('TIMEOUT')
            return {'status': 'TIMEOUT', 'expected': 0, 'detected': 0, 'output': 'Test timed out'}
        except Exception as e:
            print(f'ERROR: {e}')
            return {'status': 'ERROR', 'expected': 0, 'detected': 0, 'output': str(e)}

    def load_json_results(self):
        """Load all JSON result files"""
        json_results = {}
        if not os.path.exists(self.results_dir):
            return json_results

        for filename in os.listdir(self.results_dir):
            if filename.endswith('_output.json'):
                test_id = filename.replace('_output.json', '')
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        json_results[test_id] = data
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
        return json_results

    def run_all_tests(self):
        """Run all tests and collect results"""
        test_files = self.find_all_tests()

        print("="*80)
        print(f"RUNNING AI TESTS ({len(test_files)} total)")
        print("="*80)
        print()

        self.start_time = time.time()

        for test_file in test_files:
            test_name = os.path.basename(test_file).replace('.py', '')
            result = self.run_single_test(test_file)
            self.test_results.append({
                'test_id': test_name,
                'status': result['status'],
                'expected': result['expected'],
                'detected': result['detected'],
                'output': result['output']
            })

        self.end_time = time.time()

        print()
        print("="*80)
        print("ALL AI TESTS COMPLETED")
        print("="*80)
        print()


def main():
    """Main function"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests', 'ai_tests')
    results_dir = os.path.join(script_dir, 'results', 'ai_tests')

    # Find venv Python
    python_executable = find_venv_python()

    print()
    print("="*80)
    print("AI PEOPLE DETECTION TEST SUITE")
    print("="*80)
    print(f"Using Python: {python_executable}")
    print()

    runner = AITestRunner(tests_dir, results_dir, python_executable)
    runner.run_all_tests()

    total_duration = runner.end_time - runner.start_time

    # Load JSON results
    print("Loading test results...")
    json_results = runner.load_json_results()
    print(f"Loaded {len(json_results)} JSON result files")
    print()

    # Summary
    print("="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print()
    print(f"{'Test ID':<12} {'Status':<8} {'Expected':<10} {'Detected':<10}")
    print("-"*40)

    for result in runner.test_results:
        status_symbol = '✓' if result['status'] == 'PASS' else '✗' if result['status'] == 'FAIL' else '?'
        print(f"{result['test_id']:<12} {status_symbol} {result['status']:<5} {result['expected']:<10} {result['detected']:<10}")

    print("-"*40)
    print()

    passed = sum(1 for t in runner.test_results if t['status'] == 'PASS')
    failed = sum(1 for t in runner.test_results if t['status'] == 'FAIL')
    total = len(runner.test_results)
    pass_rate = (passed / total * 100) if total > 0 else 0
    print(f"Tests: {passed}/{total} passed ({pass_rate:.1f}%)")
    print(f"Execution Time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print()
    print(f"Results saved to: {results_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
