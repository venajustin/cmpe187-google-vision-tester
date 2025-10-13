#!/usr/bin/env python3
"""
Comprehensive Test Runner for People Detection Test Suite

This script:
1. Runs all 39 test cases (BVA, EP, DT)
2. Collects results from JSON output files
3. Generates a comprehensive summary report
4. Analyzes bugs, performance, and coverage
"""

import os
import sys
import json
import time
import subprocess
# from generate_report import ReportGenerator  # Commented out - optional report generation

class TestRunner:
    def __init__(self, tests_dir, results_dir):
        self.tests_dir = tests_dir
        self.results_dir = results_dir
        self.test_results = []
        self.start_time = None
        self.end_time = None

    def find_all_tests(self):
        """Find all test files (BVA, EP, DT)"""
        test_files = []
        for filename in sorted(os.listdir(self.tests_dir)):
            if filename.endswith('.py') and any(filename.startswith(prefix) for prefix in ['BVA-', 'EP-', 'DT-']):
                test_files.append(os.path.join(self.tests_dir, filename))
        return test_files

    def run_single_test(self, test_file):
        """Run a single test and return its result"""
        test_name = os.path.basename(test_file)
        print(f"Running {test_name}...", end=' ')
        sys.stdout.flush()

        try:
            # Run the test
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per test
            )

            # Check if test passed or failed from output
            output = result.stdout + result.stderr
            if 'PASS ✓' in output:
                print('✓ PASS')
                return {'status': 'PASS', 'output': output}
            elif 'FAIL ✗' in output:
                print('✗ FAIL')
                return {'status': 'FAIL', 'output': output}
            else:
                print('? UNKNOWN')
                return {'status': 'ERROR', 'output': output}

        except subprocess.TimeoutExpired:
            print('✗ TIMEOUT')
            return {'status': 'TIMEOUT', 'output': 'Test timed out after 120 seconds'}
        except Exception as e:
            print(f'✗ ERROR: {e}')
            return {'status': 'ERROR', 'output': str(e)}

    def load_json_results(self):
        """Load all JSON result files"""
        json_results = {}
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
        print(f"RUNNING ALL TESTS ({len(test_files)} total)")
        print("="*80)
        print()

        self.start_time = time.time()

        for test_file in test_files:
            test_name = os.path.basename(test_file).replace('.py', '')
            result = self.run_single_test(test_file)
            self.test_results.append({
                'test_id': test_name,
                'status': result['status'],
                'output': result['output']
            })

        self.end_time = time.time()

        print()
        print("="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print()

def main():
    """Main function"""
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests')
    results_dir = os.path.join(script_dir, 'results')

    print()
    print("="*80)
    print("PEOPLE DETECTION TEST SUITE - COMPREHENSIVE TEST RUNNER")
    print("="*80)
    print()

    # Run all tests
    runner = TestRunner(tests_dir, results_dir)
    runner.run_all_tests()

    total_duration = runner.end_time - runner.start_time

    # Load JSON results
    print("Loading test results...")
    json_results = runner.load_json_results()
    print(f"Loaded {len(json_results)} JSON result files")

    if len(json_results) == 0:
        print()
        print("WARNING: No JSON result files found!")
        print("This usually means:")
        print("  - Test images are missing from People_Tests/images/")
        print("  - Tests encountered errors before generating output")
        print("  - Google Cloud authentication is not set up")
        print()
        print("The report will be generated with limited data.")
        print()

    print()

    # Generate report (OPTIONAL - uncomment if generate_report.py is available)
    # print("Generating comprehensive report...")
    # generator = ReportGenerator(runner.test_results, json_results, total_duration)
    #
    # output_file = os.path.join(script_dir, 'TEST_REPORT.txt')
    # report_text = generator.generate_report(output_file)
    #
    # print()
    # print("="*80)
    # print(f"Report saved to: {output_file}")
    # print("="*80)
    # print()
    #
    # # Print summary to console
    # print("QUICK SUMMARY:")
    # print("="*80)
    # passed, failed, total, pass_rate, fail_rate = generator.calculate_pass_fail()
    # detection_rate, fp_rate, fn_rate = generator.calculate_detection_metrics()
    #
    # print(f"Tests: {passed}/{total} passed ({pass_rate:.1f}%)")
    # print(f"Detection Rate: {detection_rate:.1f}% (Target: 85%)")
    # print(f"Bugs Found: {len(generator.bugs)}")
    # print(f"Execution Time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    # print()
    #
    # meets_requirement = detection_rate >= 85
    # print(f"Meets 85% requirement: {'YES ✓' if meets_requirement else 'NO ✗'}")
    # print()
    # print(f"Full report: {output_file}")
    # print("="*80)

    # Simple summary without report generation
    print("="*80)
    print("TEST EXECUTION COMPLETE")
    print("="*80)
    passed = sum(1 for t in runner.test_results if t['status'] == 'PASS')
    total = len(runner.test_results)
    print(f"Tests: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print(f"Execution Time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print()
    print(f"Results saved to: {results_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
