"""
Farming Equipment Decision Tool - Terminal UI Version
Interactive TUI-based tool for evaluating equipment investment decisions
based on cost, milk quality improvement, and production increase.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Static, Button, Input, Label
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.align import Align


class EquipmentDecisionModel:
    """Model for calculating equipment investment decisions"""

    def __init__(self):
        self.reset_baseline()

    def reset_baseline(self):
        """Reset baseline farming parameters"""
        self.baseline_milk_production = 1000  # liters per month
        self.baseline_milk_price = 6.0  # NOK per liter
        self.baseline_quality_premium = 0.0  # percentage premium for quality

    def calculate_roi(self, equipment_cost, quality_increase_prob,
                     quality_premium_increase, production_increase_pct,
                     time_horizon_years):
        """
        Calculate Return on Investment for equipment purchase

        Parameters:
        - equipment_cost: Cost of equipment in NOK
        - quality_increase_prob: Probability (0-1) that quality improves
        - quality_premium_increase: Additional premium percentage for quality
        - production_increase_pct: Percentage increase in milk production
        - time_horizon_years: Years to evaluate ROI over

        Returns:
        - Dictionary with financial analysis
        """

        # Baseline revenue (monthly)
        baseline_monthly_revenue = (
            self.baseline_milk_production * self.baseline_milk_price *
            (1 + self.baseline_quality_premium)
        )

        # New production with equipment
        new_production = self.baseline_milk_production * (1 + production_increase_pct / 100)

        # Expected quality premium (probability-weighted)
        expected_quality_premium = (
            self.baseline_quality_premium +
            quality_increase_prob * quality_premium_increase / 100
        )

        # New monthly revenue
        new_monthly_revenue = (
            new_production * self.baseline_milk_price *
            (1 + expected_quality_premium)
        )

        # Monthly benefit
        monthly_benefit = new_monthly_revenue - baseline_monthly_revenue

        # Total benefit over time horizon
        total_months = time_horizon_years * 12
        total_benefit = monthly_benefit * total_months

        # Net benefit (profit)
        net_benefit = total_benefit - equipment_cost

        # ROI percentage
        roi_pct = (net_benefit / equipment_cost * 100) if equipment_cost > 0 else 0

        # Payback period in months
        payback_months = (equipment_cost / monthly_benefit) if monthly_benefit > 0 else float('inf')

        # Annual return
        annual_return = monthly_benefit * 12

        return {
            'baseline_monthly_revenue': baseline_monthly_revenue,
            'new_monthly_revenue': new_monthly_revenue,
            'monthly_benefit': monthly_benefit,
            'annual_return': annual_return,
            'total_benefit': total_benefit,
            'net_benefit': net_benefit,
            'roi_pct': roi_pct,
            'payback_months': payback_months,
            'equipment_cost': equipment_cost,
            'time_horizon_years': time_horizon_years
        }

    def get_recommendation(self, results):
        """Get investment recommendation based on results"""
        if results['net_benefit'] > 0 and results['payback_months'] <= 36:
            return "STRONGLY RECOMMENDED", "green"
        elif results['net_benefit'] > 0 and results['payback_months'] <= 60:
            return "RECOMMENDED", "bright_green"
        elif results['net_benefit'] > 0:
            return "MARGINAL - Long Payback", "yellow"
        else:
            return "NOT RECOMMENDED", "red"


class ParameterInput(Static):
    """Widget for parameter input with label and input field"""

    value = reactive(0.0)

    def __init__(self, label: str, min_val: float, max_val: float,
                 initial_val: float, unit: str = "", step: float = 1.0,
                 param_id: str = ""):
        super().__init__()
        self.label_text = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.unit = unit
        self.step = step
        self.param_id = param_id

    def compose(self) -> ComposeResult:
        with Horizontal(classes="parameter-row"):
            yield Label(self.label_text, classes="param-label")
            yield Input(
                value=str(self.value),
                placeholder=f"{self.min_val}-{self.max_val}",
                id=self.param_id,
                classes="param-input"
            )
            yield Label(self.unit, classes="param-unit")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        try:
            val = float(event.value)
            if self.min_val <= val <= self.max_val:
                self.value = val
                self.post_message(self.ValueChanged(self.param_id, val))
        except ValueError:
            pass

    class ValueChanged(Message):
        """Message for value changes"""
        def __init__(self, param_id: str, value: float):
            super().__init__()
            self.param_id = param_id
            self.value = value


class ResultsPanel(Static):
    """Panel displaying financial analysis results"""

    results = reactive({})

    def __init__(self):
        super().__init__()
        self.model = EquipmentDecisionModel()

    def render(self) -> Panel:
        """Render the results panel"""
        if not self.results:
            return Panel("No results yet", title="Financial Analysis", border_style="blue")

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right")

        # Add rows with formatted values
        table.add_row(
            "Monthly Revenue (Baseline)",
            f"{self.results['baseline_monthly_revenue']:,.0f} NOK"
        )
        table.add_row(
            "Monthly Revenue (New)",
            f"[blue bold]{self.results['new_monthly_revenue']:,.0f} NOK[/blue bold]"
        )
        table.add_row(
            "Monthly Benefit",
            f"[green]+{self.results['monthly_benefit']:,.0f} NOK[/green]"
            if self.results['monthly_benefit'] > 0
            else f"[red]{self.results['monthly_benefit']:,.0f} NOK[/red]"
        )
        table.add_row(
            "Annual Return",
            f"[green]{self.results['annual_return']:,.0f} NOK[/green]"
            if self.results['annual_return'] > 0
            else f"[red]{self.results['annual_return']:,.0f} NOK[/red]"
        )
        table.add_row(
            "Total Benefit",
            f"{self.results['total_benefit']:,.0f} NOK"
        )
        table.add_row(
            "Equipment Cost",
            f"[red]-{self.results['equipment_cost']:,.0f} NOK[/red]"
        )
        table.add_row(
            "Net Benefit",
            f"[green bold]{self.results['net_benefit']:,.0f} NOK[/green bold]"
            if self.results['net_benefit'] > 0
            else f"[red bold]{self.results['net_benefit']:,.0f} NOK[/red bold]"
        )
        table.add_row(
            "ROI",
            f"[bold]{self.results['roi_pct']:.1f}%[/bold]"
        )

        payback_text = (
            f"{self.results['payback_months']:.1f} months"
            if self.results['payback_months'] != float('inf')
            else "Never"
        )
        table.add_row("Payback Period", payback_text)

        # Add recommendation
        recommendation, color = self.model.get_recommendation(self.results)
        table.add_row("")  # Empty row for spacing
        table.add_row(
            "RECOMMENDATION",
            f"[{color} bold]{recommendation}[/{color} bold]"
        )

        return Panel(
            Align.center(table),
            title="Financial Analysis",
            border_style="bright_blue",
            padding=(1, 2)
        )


class EquipmentDecisionApp(App):
    """Main TUI Application"""

    CSS = """
    Screen {
        background: $surface;
    }

    #title {
        width: 100%;
        height: 3;
        content-align: center middle;
        text-style: bold;
        background: $primary;
        color: $text;
    }

    #content {
        height: 100%;
    }

    #inputs-container {
        width: 1fr;
        height: 100%;
        padding: 1 2;
    }

    #results-container {
        width: 1fr;
        height: 100%;
        padding: 1 2;
    }

    .parameter-row {
        height: 3;
        margin: 1 0;
    }

    .param-label {
        width: 30;
        content-align: left middle;
    }

    .param-input {
        width: 15;
    }

    .param-unit {
        width: 10;
        content-align: left middle;
        color: $text-muted;
    }

    #presets-container {
        margin-top: 2;
        padding: 1;
        border: solid $primary;
    }

    .preset-button {
        margin: 0 1;
    }

    #instructions {
        margin-top: 1;
        padding: 1;
        color: $text-muted;
        text-align: center;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reset", "Reset"),
    ]

    def __init__(self):
        super().__init__()
        self.model = EquipmentDecisionModel()
        self.params = {
            'equipment_cost': 100000.0,
            'quality_prob': 70.0,
            'quality_premium': 10.0,
            'production_increase': 15.0,
            'time_horizon': 5.0
        }

    def compose(self) -> ComposeResult:
        """Create child widgets"""
        yield Header(show_clock=False)

        with Container(id="content"):
            yield Static(
                "Farming Equipment Investment Decision Tool",
                id="title"
            )

            with Horizontal():
                with VerticalScroll(id="inputs-container"):
                    yield Static("[cyan bold]Parameters[/cyan bold]", classes="section-title")
                    yield ParameterInput(
                        "Equipment Cost",
                        10000, 500000, 100000,
                        " NOK", 1000, "equipment_cost"
                    )
                    yield ParameterInput(
                        "Quality Increase Probability",
                        0, 100, 70,
                        "%", 1, "quality_prob"
                    )
                    yield ParameterInput(
                        "Quality Premium Increase",
                        0, 30, 10,
                        "%", 0.5, "quality_premium"
                    )
                    yield ParameterInput(
                        "Production Increase",
                        0, 50, 15,
                        "%", 0.5, "production_increase"
                    )
                    yield ParameterInput(
                        "Time Horizon",
                        1, 15, 5,
                        " years", 1, "time_horizon"
                    )

                    with Container(id="presets-container"):
                        yield Static("[bold]Preset Scenarios:[/bold]")
                        with Horizontal():
                            yield Button("Budget", id="preset_budget",
                                       variant="success", classes="preset-button")
                            yield Button("Mid-Range", id="preset_mid",
                                       variant="primary", classes="preset-button")
                            yield Button("Premium", id="preset_premium",
                                       variant="warning", classes="preset-button")

                    yield Static(
                        "Use [b]Tab[/b] to navigate • [b]Enter[/b] to update • [b]Q[/b] to quit",
                        id="instructions"
                    )

                with Container(id="results-container"):
                    yield ResultsPanel()

        yield Footer()

    def on_mount(self) -> None:
        """Called when app is mounted"""
        self.update_results()

    def on_parameter_input_value_changed(self, message: ParameterInput.ValueChanged) -> None:
        """Handle parameter value changes"""
        self.params[message.param_id] = message.value
        self.update_results()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle preset button presses"""
        presets = {
            'preset_budget': {
                'equipment_cost': 50000,
                'quality_prob': 40,
                'quality_premium': 5,
                'production_increase': 8,
                'time_horizon': 5
            },
            'preset_mid': {
                'equipment_cost': 150000,
                'quality_prob': 70,
                'quality_premium': 12,
                'production_increase': 18,
                'time_horizon': 5
            },
            'preset_premium': {
                'equipment_cost': 350000,
                'quality_prob': 90,
                'quality_premium': 20,
                'production_increase': 30,
                'time_horizon': 5
            }
        }

        if event.button.id in presets:
            preset = presets[event.button.id]
            self.apply_preset(preset)

    def apply_preset(self, preset: dict) -> None:
        """Apply preset values to parameters"""
        for param_id, value in preset.items():
            self.params[param_id] = float(value)
            # Update input field
            input_widget = self.query_one(f"#{param_id}", Input)
            input_widget.value = str(value)

        self.update_results()

    def update_results(self) -> None:
        """Calculate and update results display"""
        results = self.model.calculate_roi(
            equipment_cost=self.params['equipment_cost'],
            quality_increase_prob=self.params['quality_prob'] / 100,
            quality_premium_increase=self.params['quality_premium'],
            production_increase_pct=self.params['production_increase'],
            time_horizon_years=self.params['time_horizon']
        )

        results_panel = self.query_one(ResultsPanel)
        results_panel.results = results

    def action_reset(self) -> None:
        """Reset to default values"""
        default_preset = {
            'equipment_cost': 100000,
            'quality_prob': 70,
            'quality_premium': 10,
            'production_increase': 15,
            'time_horizon': 5
        }
        self.apply_preset(default_preset)


def main():
    """Entry point"""
    app = EquipmentDecisionApp()
    app.run()


if __name__ == "__main__":
    main()
