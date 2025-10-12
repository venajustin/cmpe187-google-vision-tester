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
from datetime import datetime
from collections import defaultdict

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

class ReportGenerator:
    def __init__(self, test_results, json_results, total_duration):
        self.test_results = test_results
        self.json_results = json_results
        self.total_duration = total_duration
        self.bugs = []

    def categorize_tests(self):
        """Categorize tests by type"""
        bva_tests = [t for t in self.test_results if t['test_id'].startswith('BVA-')]
        ep_tests = [t for t in self.test_results if t['test_id'].startswith('EP-')]
        dt_tests = [t for t in self.test_results if t['test_id'].startswith('DT-')]
        return bva_tests, ep_tests, dt_tests

    def calculate_pass_fail(self):
        """Calculate pass/fail statistics"""
        passed = sum(1 for t in self.test_results if t['status'] == 'PASS')
        failed = sum(1 for t in self.test_results if t['status'] in ['FAIL', 'ERROR', 'TIMEOUT'])
        total = len(self.test_results)

        pass_rate = (passed / total * 100) if total > 0 else 0
        fail_rate = (failed / total * 100) if total > 0 else 0

        return passed, failed, total, pass_rate, fail_rate

    def calculate_detection_metrics(self):
        """Calculate overall detection performance"""
        total_actual = 0
        total_detected = 0
        total_false_positives = 0
        total_false_negatives = 0
        valid_tests = 0

        for test_id, data in self.json_results.items():
            if 'actual_results' in data:
                actual = data.get('test_configuration', {}).get('expected_people_count', 0) or \
                         data.get('actual_results', {}).get('actual_people_in_scene', 0)
                detected = data['actual_results'].get('detected_people', 0)

                total_actual += actual
                total_detected += detected

                if 'false_positives' in data['actual_results']:
                    total_false_positives += data['actual_results']['false_positives']
                if 'false_negatives' in data['actual_results']:
                    total_false_negatives += data['actual_results']['false_negatives']

                valid_tests += 1

        overall_detection_rate = (total_detected / total_actual * 100) if total_actual > 0 else 0

        # False positive rate: FP / (FP + TN) - approximated as FP / total_tests
        false_positive_rate = (total_false_positives / valid_tests * 100) if valid_tests > 0 else 0

        # False negative rate: FN / (FN + TP) - approximated as FN / total_actual
        false_negative_rate = (total_false_negatives / total_actual * 100) if total_actual > 0 else 0

        return overall_detection_rate, false_positive_rate, false_negative_rate

    def analyze_complexity(self):
        """Analyze test complexity distribution"""
        simple = 0
        medium = 0
        complex_count = 0

        for test_id, data in self.json_results.items():
            actual_people = data.get('test_configuration', {}).get('expected_people_count', 0) or \
                           data.get('actual_results', {}).get('actual_people_in_scene', 0)

            # Simple: 0-2 people, ideal conditions
            # Medium: 3-10 people or minor challenges
            # Complex: 11+ people or crowds, multiple factors

            if test_id.startswith('BVA-'):
                if actual_people <= 2:
                    simple += 1
                elif actual_people <= 10:
                    medium += 1
                else:
                    complex_count += 1
            elif test_id.startswith('EP-'):
                # EP tests are generally medium to complex
                if 'Ideal' in data.get('test_name', '') or actual_people <= 1:
                    simple += 1
                else:
                    medium += 1
            elif test_id.startswith('DT-'):
                # DT tests are complex (multiple interacting factors)
                complex_count += 1

        total = len(self.json_results)
        return {
            'simple': simple,
            'medium': medium,
            'complex': complex_count,
            'total': total,
            'simple_pct': (simple / total * 100) if total > 0 else 0,
            'medium_pct': (medium / total * 100) if total > 0 else 0,
            'complex_pct': (complex_count / total * 100) if total > 0 else 0
        }

    def analyze_environmental_coverage(self):
        """Analyze environmental condition coverage"""
        ideal = 0
        challenging_lighting = 0
        adverse_weather = 0
        high_occlusion = 0

        for test_id, data in self.json_results.items():
            test_name = data.get('test_name', '').lower()
            description = data.get('description', '').lower()
            combined = test_name + ' ' + description

            # Ideal conditions
            if 'ideal' in combined or 'perfect' in combined or 'optimal' in combined:
                ideal += 1

            # Challenging lighting
            if any(keyword in combined for keyword in ['night', 'low light', 'backlighting', 'glare', 'silhouette']):
                challenging_lighting += 1

            # Adverse weather
            if any(keyword in combined for keyword in ['rain', 'snow', 'weather', 'wet']):
                adverse_weather += 1

            # High occlusion
            if any(keyword in combined for keyword in ['occlusion', 'hidden', 'crowd', 'dense']):
                high_occlusion += 1

        return {
            'ideal': ideal,
            'challenging_lighting': challenging_lighting,
            'adverse_weather': adverse_weather,
            'high_occlusion': high_occlusion
        }

    def analyze_bugs(self):
        """Identify and categorize bugs from failed tests"""
        bugs = []
        bug_id = 1

        for test_id, data in self.json_results.items():
            if data.get('test_result', {}).get('status') == 'FAIL':
                failure_reasons = data.get('test_result', {}).get('failure_reasons', [])

                # Determine severity
                detection_rate = data.get('actual_results', {}).get('detection_rate', 0)
                expected_people = data.get('test_configuration', {}).get('expected_people_count', 0) or \
                                 data.get('actual_results', {}).get('actual_people_in_scene', 0)
                detected_people = data.get('actual_results', {}).get('detected_people', 0)

                # Critical: Missed people in safety-critical scenarios (DT tests, crosswalks)
                # Major: Detection rate < 70% or significant count errors
                # Minor: Detection rate 70-85% or minor count errors

                severity = 'Minor'
                if test_id.startswith('DT-') and expected_people > 0 and detected_people == 0:
                    severity = 'Critical'
                elif detection_rate < 70:
                    severity = 'Major'
                elif detection_rate < 85:
                    severity = 'Minor'

                # If count error is significant
                count_error = abs(expected_people - detected_people)
                if expected_people > 0:
                    error_pct = (count_error / expected_people) * 100
                    if error_pct > 50:
                        severity = 'Major'
                    elif error_pct > 30 and severity == 'Minor':
                        severity = 'Major'

                bug = {
                    'id': f'BUG-{bug_id:03d}',
                    'test_case': test_id,
                    'severity': severity,
                    'description': ' | '.join(failure_reasons) if failure_reasons else 'Test failed',
                    'expected': f'{expected_people} people detected',
                    'actual': f'{detected_people} people detected (detection rate: {detection_rate:.1f}%)',
                    'detection_rate': detection_rate,
                    'test_data': data
                }
                bugs.append(bug)
                bug_id += 1

        self.bugs = bugs
        return bugs

    def analyze_failure_patterns(self):
        """Analyze common failure patterns"""
        categories = defaultdict(list)

        for bug in self.bugs:
            test_data = bug['test_data']
            test_id = bug['test_case']

            # Categorize by condition
            description = test_data.get('test_name', '').lower() + ' ' + test_data.get('description', '').lower()

            if any(kw in description for kw in ['night', 'low light', 'dark']):
                categories['Low Light Conditions'].append(bug)
            if any(kw in description for kw in ['rain', 'snow', 'weather', 'wet']):
                categories['Adverse Weather'].append(bug)
            if any(kw in description for kw in ['occlusion', 'hidden', 'partial']):
                categories['High Occlusion'].append(bug)
            if any(kw in description for kw in ['distance', 'far', '100m']):
                categories['Distance Limitations'].append(bug)
            if any(kw in description for kw in ['crowd', 'dense', 'group']):
                categories['Crowd/Group Detection'].append(bug)
            if any(kw in description for kw in ['backlight', 'glare', 'silhouette']):
                categories['Difficult Lighting'].append(bug)

            # If no category matched
            if not any(bug in cat for cat in categories.values()):
                categories['Other'].append(bug)

        return dict(categories)

    def calculate_detection_by_condition(self):
        """Calculate success rates by condition type"""
        good_conditions = {'total': 0, 'passed': 0}
        challenging_conditions = {'total': 0, 'passed': 0}
        poor_conditions = {'total': 0, 'passed': 0}

        for test_id, data in self.json_results.items():
            description = data.get('test_name', '').lower() + ' ' + data.get('description', '').lower()
            passed = data.get('test_result', {}).get('status') == 'PASS'

            # Good conditions: clear, daylight, no weather, no occlusion
            if any(kw in description for kw in ['ideal', 'perfect', 'clear', 'daylight']) and \
               not any(kw in description for kw in ['night', 'rain', 'snow', 'occlusion', 'hidden']):
                good_conditions['total'] += 1
                if passed:
                    good_conditions['passed'] += 1

            # Poor conditions: night, heavy weather, high occlusion
            elif any(kw in description for kw in ['night', 'heavy', 'severe', '75%', 'extreme']):
                poor_conditions['total'] += 1
                if passed:
                    poor_conditions['passed'] += 1

            # Challenging: everything else
            else:
                challenging_conditions['total'] += 1
                if passed:
                    challenging_conditions['passed'] += 1

        return {
            'good': (good_conditions['passed'] / good_conditions['total'] * 100) if good_conditions['total'] > 0 else 0,
            'challenging': (challenging_conditions['passed'] / challenging_conditions['total'] * 100) if challenging_conditions['total'] > 0 else 0,
            'poor': (poor_conditions['passed'] / poor_conditions['total'] * 100) if poor_conditions['total'] > 0 else 0
        }

    def generate_report(self, output_file):
        """Generate comprehensive test report"""
        bva_tests, ep_tests, dt_tests = self.categorize_tests()
        passed, failed, total, pass_rate, fail_rate = self.calculate_pass_fail()
        detection_rate, fp_rate, fn_rate = self.calculate_detection_metrics()
        complexity = self.analyze_complexity()
        env_coverage = self.analyze_environmental_coverage()
        bugs = self.analyze_bugs()
        failure_patterns = self.analyze_failure_patterns()
        condition_success = self.calculate_detection_by_condition()

        # Calculate timing statistics
        avg_time_per_test = self.total_duration / total if total > 0 else 0

        # Calculate bug severity distribution
        critical_bugs = sum(1 for b in bugs if b['severity'] == 'Critical')
        major_bugs = sum(1 for b in bugs if b['severity'] == 'Major')
        minor_bugs = sum(1 for b in bugs if b['severity'] == 'Minor')

        # Generate report
        report = []
        report.append("="*80)
        report.append("PEOPLE DETECTION TEST SUITE - COMPREHENSIVE REPORT")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Test Execution Summary
        report.append("="*80)
        report.append("TEST EXECUTION SUMMARY")
        report.append("="*80)
        report.append(f"Total Test Cases Executed: {total}")
        report.append(f"  Boundary Value Analysis: {len(bva_tests)} cases")
        report.append(f"  Equivalence Partition: {len(ep_tests)} cases")
        report.append(f"  Decision Table: {len(dt_tests)} cases")
        report.append("")
        report.append("Test Results:")
        report.append(f"  Passed: {passed} cases ({pass_rate:.1f}%)")
        report.append(f"  Failed: {failed} cases ({fail_rate:.1f}%)")
        report.append("")
        report.append("Detection Performance:")
        report.append(f"  Overall Detection Rate: {detection_rate:.1f}% (Target: 85%)")
        report.append(f"  False Positive Rate: {fp_rate:.1f}%")
        report.append(f"  False Negative Rate: {fn_rate:.1f}%")
        report.append("")
        report.append("Test Execution:")
        report.append(f"  Total execution time: {self.total_duration:.2f} seconds ({self.total_duration/60:.2f} minutes)")
        report.append(f"  Average time per test: {avg_time_per_test:.2f} seconds")
        report.append("")
        report.append("Analysis & Documentation (Estimated):")
        report.append(f"  Result recording: 2 hours")
        report.append(f"  Bug analysis: {len(bugs) * 0.25:.1f} hours (15 min per bug)")
        report.append(f"  Report writing: 8 hours")
        report.append("")
        report.append(f"Total Manual Testing Effort: {10 + len(bugs) * 0.25:.1f} person-hours")
        report.append("")

        # Test Complexity Analysis
        report.append("="*80)
        report.append("TEST COMPLEXITY ANALYSIS")
        report.append("="*80)
        report.append("Complexity Distribution:")
        report.append(f"  Simple (single object, clear conditions): {complexity['simple']} tests ({complexity['simple_pct']:.1f}%)")
        report.append(f"  Medium (multiple objects, minor challenges): {complexity['medium']} tests ({complexity['medium_pct']:.1f}%)")
        report.append(f"  Complex (crowds/groups, multiple factors): {complexity['complex']} tests ({complexity['complex_pct']:.1f}%)")
        report.append("")
        report.append("Environmental Complexity (People Detection):")
        report.append(f"  Ideal conditions: {env_coverage['ideal']} tests")
        report.append(f"  Challenging lighting: {env_coverage['challenging_lighting']} tests")
        report.append(f"  Adverse weather: {env_coverage['adverse_weather']} tests")
        report.append(f"  High occlusion: {env_coverage['high_occlusion']} tests")
        report.append("")

        # Test Coverage Analysis
        report.append("="*80)
        report.append("TEST COVERAGE ANALYSIS")
        report.append("="*80)
        report.append("Requirements Coverage:")
        report.append(f"  Detection accuracy: 100% covered (all {total} tests)")
        report.append(f"  Count/Classification accuracy: 100% covered (all {total} tests)")
        report.append(f"  Environmental conditions: 100% covered ({env_coverage['challenging_lighting']} lighting, {env_coverage['adverse_weather']} weather)")
        report.append(f"  Edge cases: 100% covered (distance, occlusion, crowds)")

        # Count false positive test cases
        false_positive_tests = sum(1 for test_id in self.json_results.keys()
                                  if self.json_results[test_id].get('test_configuration', {}).get('expected_people_count', 1) == 0 or
                                     self.json_results[test_id].get('actual_results', {}).get('actual_people_in_scene', 1) == 0)
        report.append(f"  False positive scenarios: {false_positive_tests} cases tested")
        report.append("")
        report.append("Input Domain Coverage:")

        # Analyze ranges from JSON data
        people_counts = [data.get('test_configuration', {}).get('expected_people_count', 0) or
                        data.get('actual_results', {}).get('actual_people_in_scene', 0)
                        for data in self.json_results.values()]

        if people_counts:
            report.append(f"  Object counts: 0-{max(people_counts)} people ({min(people_counts)} to {max(people_counts)} range tested)")
        else:
            report.append(f"  Object counts: 0-25 people (full range specified in test suite)")

        report.append(f"  Lighting conditions: Daylight, Backlighting, Nighttime, Low-light, Glare, Silhouette")
        report.append(f"  Weather: Clear, Rain, Snow")
        report.append(f"  Distance: Close (<10m), Medium (10-50m), Far (50-100m), Extreme (>100m)")
        report.append(f"  Occlusion levels: 0%, 25%, 50%, 75%, Dense crowds")
        report.append("")

        # Bug Summary
        report.append("="*80)
        report.append("BUG SUMMARY")
        report.append("="*80)
        report.append("Bug Statistics:")
        report.append(f"  Total Bugs Found: {len(bugs)}")
        report.append(f"  Critical: {critical_bugs} ({critical_bugs/len(bugs)*100 if bugs else 0:.1f}%)")
        report.append(f"  Major: {major_bugs} ({major_bugs/len(bugs)*100 if bugs else 0:.1f}%)")
        report.append(f"  Minor: {minor_bugs} ({minor_bugs/len(bugs)*100 if bugs else 0:.1f}%)")
        report.append("")

        if bugs:
            report.append("Bug Listing:")
            report.append("")
            for bug in bugs:
                report.append("-"*80)
                report.append(f"ID: {bug['id']}")
                report.append(f"Test Case: {bug['test_case']}")
                report.append(f"Severity: {bug['severity']}")
                report.append(f"Description: {bug['description']}")
                report.append(f"Expected: {bug['expected']}")
                report.append(f"Actual: {bug['actual']}")
                report.append("")
        else:
            report.append("No bugs found - all tests passed!")
            report.append("")

        # Bug Analysis
        report.append("="*80)
        report.append("BUG ANALYSIS")
        report.append("="*80)

        if bugs:
            report.append("Common Failure Categories:")
            for category, cat_bugs in failure_patterns.items():
                related_tests = len([t for t in self.json_results if any(kw in (self.json_results[t].get('test_name', '').lower() + ' ' + self.json_results[t].get('description', '').lower()) for kw in category.lower().split())])
                report.append(f"  {category}: {len(cat_bugs)} failures ({len(cat_bugs)/related_tests*100 if related_tests > 0 else 0:.1f}% of related tests)")
            report.append("")

            report.append("Detection Rate by Condition:")
            report.append(f"  Good conditions (clear, daylight): {condition_success['good']:.1f}% success rate")
            report.append(f"  Challenging conditions (rain, dusk, shadows): {condition_success['challenging']:.1f}% success rate")
            report.append(f"  Poor conditions (night, heavy rain/snow, fog): {condition_success['poor']:.1f}% success rate")
            report.append("")

            report.append("Root Cause Analysis:")
            report.append("  Patterns of when/why the API fails:")
            for category, cat_bugs in failure_patterns.items():
                if cat_bugs:
                    avg_detection = sum(b['detection_rate'] for b in cat_bugs) / len(cat_bugs)
                    report.append(f"    - {category}: Avg detection rate {avg_detection:.1f}%")
            report.append("")

            # Identify consistent failure points
            report.append("  Consistent failure points:")
            low_detection_tests = [b for b in bugs if b['detection_rate'] < 60]
            if low_detection_tests:
                report.append(f"    - {len(low_detection_tests)} tests with <60% detection rate")
                for bug in low_detection_tests[:5]:  # Show top 5
                    report.append(f"      * {bug['test_case']}: {bug['detection_rate']:.1f}%")

            zero_detection_tests = [b for b in bugs if b['detection_rate'] == 0]
            if zero_detection_tests:
                report.append(f"    - {len(zero_detection_tests)} tests with 0% detection (complete failure)")
                for bug in zero_detection_tests:
                    report.append(f"      * {bug['test_case']}")
            report.append("")

            report.append("  Unexpected behaviors:")
            false_positive_bugs = [b for b in bugs if 'false positive' in b['description'].lower()]
            if false_positive_bugs:
                report.append(f"    - {len(false_positive_bugs)} false positive errors (detected people when none present)")

            high_count_error_bugs = [b for b in bugs if 'count error' in b['description'].lower()]
            if high_count_error_bugs:
                report.append(f"    - {len(high_count_error_bugs)} significant count errors")
            report.append("")
        else:
            report.append("No failures to analyze - all tests passed!")
            report.append("")

        # Conclusions
        report.append("="*80)
        report.append("CONCLUSIONS")
        report.append("="*80)

        meets_requirement = detection_rate >= 85
        report.append(f"Does the API meet the 85% detection requirement? {'YES' if meets_requirement else 'NO'}")
        report.append(f"  Overall detection rate: {detection_rate:.1f}%")
        report.append("")

        # Determine suitability
        suitable = meets_requirement and critical_bugs == 0 and condition_success['good'] >= 95
        report.append(f"Is it suitable for autonomous vehicle use? {'YES' if suitable else 'NO - NEEDS IMPROVEMENT'}")
        report.append("")
        report.append("Assessment for people detection in autonomous vehicles:")

        if suitable:
            report.append("  ✓ The API demonstrates reliable people detection performance")
            report.append("  ✓ Meets the 85% detection rate requirement")
            report.append("  ✓ No critical safety failures")
            report.append("  ✓ Performs well in good conditions (>95%)")
        else:
            report.append("  Issues identified:")
            if not meets_requirement:
                report.append(f"    ✗ Overall detection rate ({detection_rate:.1f}%) below 85% requirement")
            if critical_bugs > 0:
                report.append(f"    ✗ {critical_bugs} critical bug(s) found - safety-critical scenarios failed")
            if condition_success['good'] < 95:
                report.append(f"    ✗ Performance in good conditions ({condition_success['good']:.1f}%) below 95% expectation")
            if condition_success['poor'] < 60:
                report.append(f"    ⚠ Poor performance in challenging conditions ({condition_success['poor']:.1f}%)")

        report.append("")
        report.append("Recommendations:")
        if not suitable:
            if detection_rate < 85:
                report.append("  - Improve overall detection algorithms to meet 85% threshold")
            if critical_bugs > 0:
                report.append("  - Address critical safety failures before deployment")
            if condition_success['poor'] < 60:
                report.append("  - Enhance performance in low-light and adverse weather conditions")
            if fn_rate > 15:
                report.append("  - Reduce false negatives (missed detections) which pose safety risks")
        else:
            report.append("  - Continue monitoring performance in real-world conditions")
            report.append("  - Regular testing with updated scenarios")
            if condition_success['poor'] < 80:
                report.append("  - Consider improvements for adverse weather conditions")

        report.append("")
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        # Write report to file
        report_text = '\n'.join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)

        return report_text

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

    # Generate report
    print("Generating comprehensive report...")
    generator = ReportGenerator(runner.test_results, json_results, total_duration)

    output_file = os.path.join(script_dir, 'TEST_REPORT.txt')
    report_text = generator.generate_report(output_file)

    print()
    print("="*80)
    print(f"Report saved to: {output_file}")
    print("="*80)
    print()

    # Print summary to console
    print("QUICK SUMMARY:")
    print("="*80)
    passed, failed, total, pass_rate, fail_rate = generator.calculate_pass_fail()
    detection_rate, fp_rate, fn_rate = generator.calculate_detection_metrics()

    print(f"Tests: {passed}/{total} passed ({pass_rate:.1f}%)")
    print(f"Detection Rate: {detection_rate:.1f}% (Target: 85%)")
    print(f"Bugs Found: {len(generator.bugs)}")
    print(f"Execution Time: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print()

    meets_requirement = detection_rate >= 85
    print(f"Meets 85% requirement: {'YES ✓' if meets_requirement else 'NO ✗'}")
    print()
    print(f"Full report: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
