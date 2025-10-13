# Farming Equipment Investment Decision Tool

An interactive terminal UI (TUI) tool for evaluating farming equipment investments based on cost, milk quality improvements, and production increases. This tool helps farmers make data-driven decisions about equipment purchases by calculating ROI, payback periods, and expected returns.

Built with [Textual](https://textual.textualize.io/) - a modern Python framework for building sophisticated terminal user interfaces.

## Overview

This tool addresses a common farming decision problem: **Should I invest in cheaper or more expensive equipment?**

The answer depends on several factors:
- Initial equipment cost
- Probability of milk quality improvement
- Expected quality premium increase (higher quality = higher price)
- Expected production increase (more milk produced)
- Time horizon for the investment

## Features

- **Terminal-based Interface**: Runs directly in your terminal - no GUI required
- **Interactive Input Fields**: Edit parameters with keyboard navigation
- **Financial Analysis**: Comprehensive ROI calculations including:
  - Monthly and annual revenue projections
  - Net benefit calculations
  - Payback period estimation
  - ROI percentage
- **Color-coded Recommendations**: Visual feedback based on financial viability
- **Preset Scenarios**: Quick comparison of budget, mid-range, and premium equipment
- **Real-time Updates**: All calculations update instantly as you adjust parameters
- **Keyboard Shortcuts**: Navigate efficiently with Tab, Enter, and hotkeys

## Installation

### Prerequisites
- Python 3.12+ (included in this project's virtual environment)
- textual and rich libraries (for terminal UI)

### Setup

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r project_ideas/farming_equipment_decision/requirements.txt
```

## Usage

### Running the Tool

From the project root directory:

```bash
source .venv/bin/activate
python3 project_ideas/farming_equipment_decision/equipment_decision_tui.py
```

The application will launch in your terminal with a full-screen interactive interface.

### Interface Guide

#### Keyboard Navigation:

- **Tab**: Move between input fields
- **Enter**: Confirm input value and update calculations
- **Q**: Quit application
- **R**: Reset to default values
- **Click buttons**: Use preset scenarios

#### Adjustable Parameters (Input Fields):

1. **Equipment Cost** (10,000 - 500,000 NOK)
   - The purchase price of the equipment
   - Includes installation and setup costs

2. **Probability of Quality Increase** (0% - 100%)
   - Likelihood that the equipment will improve milk quality
   - Based on manufacturer claims, research, or expert opinion

3. **Quality Premium Increase** (0% - 30%)
   - Additional percentage premium you can charge for higher quality milk
   - Example: 10% means 6.00 NOK/liter becomes 6.60 NOK/liter

4. **Production Increase** (0% - 50%)
   - Percentage increase in milk production volume
   - Due to improved efficiency, automation, or better cow health

5. **Time Horizon** (1 - 15 years)
   - Investment evaluation period
   - How long you plan to use the equipment

#### Preset Scenarios:

- **Budget Equipment**: Low cost, moderate improvements
  - Cost: 50,000 NOK
  - Quality probability: 40%
  - Quality premium: 5%
  - Production increase: 8%

- **Mid-Range Equipment**: Balanced cost and performance
  - Cost: 150,000 NOK
  - Quality probability: 70%
  - Quality premium: 12%
  - Production increase: 18%

- **Premium Equipment**: High cost, maximum improvements
  - Cost: 350,000 NOK
  - Quality probability: 90%
  - Quality premium: 20%
  - Production increase: 30%

### Financial Analysis Output

The tool calculates and displays:

- **Monthly Revenue (Baseline)**: Current monthly income
- **Monthly Revenue (New)**: Projected monthly income with new equipment
- **Monthly Benefit**: Additional monthly income
- **Annual Return**: Total additional income per year
- **Total Benefit**: Cumulative benefit over time horizon
- **Equipment Cost**: Initial investment
- **Net Benefit**: Total profit (Total Benefit - Equipment Cost)
- **ROI**: Return on Investment percentage
- **Payback Period**: Months until equipment pays for itself

### Recommendation System

The tool provides color-coded recommendations:

- **Green (STRONGLY RECOMMENDED)**: Positive ROI with payback ≤ 3 years
- **Light Green (RECOMMENDED)**: Positive ROI with payback ≤ 5 years
- **Orange (MARGINAL)**: Positive ROI but long payback period
- **Red (NOT RECOMMENDED)**: Negative ROI

## Decision Model

### Baseline Assumptions

- Baseline milk production: 1,000 liters/month
- Baseline milk price: 6.00 NOK/liter
- No current quality premium

### Calculation Method

The model uses expected value calculations to account for uncertainty:

1. **Expected Quality Premium**:
   ```
   Expected Premium = Baseline Premium + (Probability × Premium Increase)
   ```

2. **New Production**:
   ```
   New Production = Baseline Production × (1 + Production Increase %)
   ```

3. **New Revenue**:
   ```
   New Revenue = New Production × Price × (1 + Expected Premium)
   ```

4. **Net Benefit**:
   ```
   Net Benefit = (Monthly Benefit × Months) - Equipment Cost
   ```

5. **ROI**:
   ```
   ROI % = (Net Benefit / Equipment Cost) × 100
   ```

## Example Use Cases

### Case 1: Evaluating Automatic Milking System

**Scenario**: Considering a 250,000 NOK robotic milking system

**Parameters**:
- Equipment Cost: 250,000 NOK
- Quality Increase Probability: 80% (proven technology)
- Quality Premium Increase: 15% (cleaner milk, better handling)
- Production Increase: 20% (more frequent milking)
- Time Horizon: 8 years

**Result**: The tool will calculate if this investment makes financial sense

### Case 2: Comparing Two Options

**Option A - Budget**:
- Cost: 80,000 NOK
- Lower improvements but faster payback

**Option B - Premium**:
- Cost: 300,000 NOK
- Higher improvements but longer payback

Use the preset buttons and adjust parameters to find the breakeven points.

### Case 3: Sensitivity Analysis

Ask questions like:
- "How much quality improvement do I need to justify this cost?"
- "What if production only increases by 10% instead of 20%?"
- "Should I go for a longer time horizon?"

Adjust input values to see how changes affect the recommendation in real-time.

## Technical Details

### Model Class: `EquipmentDecisionModel`

Located in `equipment_decision_tui.py:17`

Key method:
```python
calculate_roi(equipment_cost, quality_increase_prob,
              quality_premium_increase, production_increase_pct,
              time_horizon_years)
```

Returns dictionary with all financial metrics.

### ParameterInput Widget: `ParameterInput`

Custom Textual widget for parameter input fields with validation.

### ResultsPanel Widget: `ResultsPanel`

Dynamic panel that updates with financial analysis using Rich tables.

### Main Application: `EquipmentDecisionApp`

Textual app that coordinates the UI, event handling, and real-time calculations.

## Customization

### Adjusting Baseline Values

Edit the `reset_baseline()` method in `EquipmentDecisionModel` (equipment_decision_tui.py:23):

```python
def reset_baseline(self):
    self.baseline_milk_production = 1000  # Your current production
    self.baseline_milk_price = 6.0        # Your current price
    self.baseline_quality_premium = 0.0   # Your current premium
```

### Adding New Presets

Add to the `presets` dictionary in the `on_button_pressed()` method (equipment_decision_tui.py:289).

### Changing Input Ranges

Modify the `min_val` and `max_val` parameters when creating `ParameterInput` widgets in the `compose()` method.

## Future Enhancements

Potential additions:
- Maintenance costs over time
- Depreciation calculations
- Multiple equipment comparison side-by-side
- Export results to PDF/CSV
- Historical data import
- Inflation adjustment
- Risk analysis with probability distributions

## License

Part of the ADA511 course materials at HVL.

## Contact

For questions or improvements, please contact the course instructors.
