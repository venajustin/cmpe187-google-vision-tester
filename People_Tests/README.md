# People Detection Test Suite for Google Cloud Vision API

Comprehensive testing framework for evaluating Google Cloud Vision API's Object Localization endpoint for pedestrian detection in autonomous vehicle scenarios.

## Quick Start

### 1. Add Test Images
Place 39 test images in the `images/` folder:
```
images/
├── BVA-001.jpg through BVA-021.jpg (21 images)
├── EP-001.jpg through EP-011.jpg (11 images)
└── DT-001.jpg through DT-007.jpg (7 images)
```

### 2. Set Up Authentication
```bash
gcloud auth application-default login
```

### 3. Run All Tests
```bash
# From the project root (cmpe187-google-vision-tester)
python3 People_Tests/run_all_tests.py

# OR from People_Tests folder
cd People_Tests
python3 run_all_tests.py
```

### 4. View Results
```bash
# View individual test outputs
ls People_Tests/results/
```

## What Gets Generated

### Individual Test Results (in results/ folder)
For each test (e.g., BVA-001):
- `BVA-001_result.jpg` - Annotated image with bounding boxes
- `BVA-001_output.json` - Structured data with all metrics and timing

## Test Suite Overview

**Total Tests**: 39
- **BVA (Boundary Value Analysis)**: 21 tests
  - Count boundaries (0-21 people)
  - Special conditions (distance, occlusion, night)

- **EP (Equivalence Partition)**: 11 tests
  - Lighting conditions
  - Weather conditions
  - False positive prevention
  - Pose variations

- **DT (Decision Table)**: 7 tests
  - Real-world safety scenarios
  - Crosswalk situations
  - Emergency scenes
  - Complex multi-factor scenarios

## Running Individual Tests

```bash
# Run single test
python3 tests/BVA-001.py

# Run by category
for test in tests/BVA-*.py; do python3 "$test"; done  # All BVA tests
for test in tests/EP-*.py; do python3 "$test"; done   # All EP tests
for test in tests/DT-*.py; do python3 "$test"; done   # All DT tests
```

## Project Structure

```
People_Tests/
├── run_all_tests.py           # Main test runner
├── images/                    # Test images (you need to add these)
│   ├── BVA-001.jpg
│   ├── BVA-002.jpg
│   └── ... (39 total, supports .jpg/.jpeg/.png)
├── results/                   # Generated test outputs (gitignored)
│   ├── BVA-001_result.jpg    # Annotated images
│   ├── BVA-001_output.json   # JSON results with timing
│   └── ...
└── tests/                     # Test scripts
    ├── BVA-001.py through BVA-021.py (21 tests)
    ├── EP-001.py through EP-011.py (11 tests)
    ├── DT-001.py through DT-007.py (7 tests)
    └── test_utils.py         # Shared utilities
```

## Key Features

### Automated Testing
- Runs all 39 tests automatically
- Tracks timing (start, end, duration) for each test
- Handles errors and timeouts gracefully
- Generates professional report

### Comprehensive Analysis
- Detection rate calculations
- False positive/negative tracking
- Bug identification and categorization (Critical/Major/Minor)
- Pattern analysis in failures
- Condition-based performance metrics


## Requirements

### Software
- Python 3.7+
- Google Cloud SDK (gcloud CLI)
- Google Cloud Vision API enabled

### Python Packages
```bash
pip install google-cloud-vision matplotlib
```

### Google Cloud Setup
1. Create or use existing Google Cloud project
2. Enable Vision API
3. Authenticate:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```


## Troubleshooting

### "Image file not found"
- **Issue**: Test images missing
- **Solution**: Add images to `People_Tests/images/` folder with correct naming (BVA-001.jpg, EP-001.jpg, DT-001.jpg, etc.)

### Authentication Errors
- **Issue**: Google Cloud credentials not found
- **Solution**: Run `gcloud auth application-default login`

### "can't open file"
- **Issue**: Running from wrong directory
- **Solution**: Use correct path:
  ```bash
  # From project root
  python3 People_Tests/run_all_tests.py

  # OR change to People_Tests folder first
  cd People_Tests
  python3 run_all_tests.py
  ```

### Test Timeouts
- **Issue**: Tests taking too long
- **Solution**: Check internet connection and Google Cloud API status

## Test Timing

All tests now include timing information:
- **Start time**: When test execution began
- **End time**: When test completed
- **Duration**: Total execution time (formatted as milliseconds, seconds, or minutes)

This helps identify slow tests and track API response times.

## Support

For issues or questions:
1. Verify Google Cloud authentication
2. Ensure test images are present and correctly named (.jpg, .jpeg, or .png)
3. Review individual test outputs in `results/` folder

