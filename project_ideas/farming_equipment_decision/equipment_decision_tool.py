"""
Farming Equipment Decision Tool
Interactive pygame-based tool for evaluating equipment investment decisions
based on cost, milk quality improvement, and production increase.
"""

import pygame
import sys
import math

# Initialize pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_GRAY = (240, 240, 240)
BLUE = (70, 130, 180)
LIGHT_BLUE = (135, 206, 250)
GREEN = (46, 125, 50)
LIGHT_GREEN = (129, 199, 132)
RED = (198, 40, 40)
LIGHT_RED = (239, 154, 154)
ORANGE = (255, 152, 0)

# Font sizes
TITLE_FONT = pygame.font.Font(None, 48)
HEADER_FONT = pygame.font.Font(None, 32)
LABEL_FONT = pygame.font.Font(None, 24)
VALUE_FONT = pygame.font.Font(None, 28)
SMALL_FONT = pygame.font.Font(None, 20)


class Slider:
    """Interactive slider for adjusting parameters"""

    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label, unit=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial_val
        self.label = label
        self.unit = unit
        self.dragging = False
        self.handle_radius = 12

    def get_handle_pos(self):
        """Calculate handle position based on current value"""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        x = self.rect.x + ratio * self.rect.width
        y = self.rect.centery
        return (int(x), int(y))

    def handle_event(self, event):
        """Handle mouse events for dragging"""
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_x, handle_y = self.get_handle_pos()
            distance = math.sqrt((event.pos[0] - handle_x)**2 + (event.pos[1] - handle_y)**2)
            if distance <= self.handle_radius * 1.5:
                self.dragging = True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False

        elif event.type == pygame.MOUSEMOTION and self.dragging:
            # Update value based on mouse position
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
            self.value = self.min_val + ratio * (self.max_val - self.min_val)

    def draw(self, surface):
        """Draw the slider"""
        # Draw track
        pygame.draw.rect(surface, GRAY, self.rect, border_radius=5)

        # Draw filled portion
        handle_x, _ = self.get_handle_pos()
        filled_rect = pygame.Rect(self.rect.x, self.rect.y,
                                  handle_x - self.rect.x, self.rect.height)
        pygame.draw.rect(surface, BLUE, filled_rect, border_radius=5)

        # Draw handle
        handle_pos = self.get_handle_pos()
        pygame.draw.circle(surface, LIGHT_BLUE if self.dragging else BLUE,
                          handle_pos, self.handle_radius)
        pygame.draw.circle(surface, DARK_GRAY, handle_pos, self.handle_radius, 2)

        # Draw label
        label_surf = LABEL_FONT.render(self.label, True, BLACK)
        surface.blit(label_surf, (self.rect.x, self.rect.y - 30))

        # Draw value
        value_text = f"{self.value:.1f}{self.unit}"
        value_surf = VALUE_FONT.render(value_text, True, BLUE)
        surface.blit(value_surf, (self.rect.right + 20, self.rect.y - 5))


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
            return "STRONGLY RECOMMENDED", GREEN
        elif results['net_benefit'] > 0 and results['payback_months'] <= 60:
            return "RECOMMENDED", LIGHT_GREEN
        elif results['net_benefit'] > 0:
            return "MARGINAL - Long Payback", ORANGE
        else:
            return "NOT RECOMMENDED", RED


class EquipmentDecisionTool:
    """Main application class"""

    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Farming Equipment Investment Decision Tool")
        self.clock = pygame.time.Clock()
        self.running = True

        self.model = EquipmentDecisionModel()

        # Create sliders
        slider_x = 100
        slider_width = 400
        slider_height = 10
        spacing = 90
        start_y = 150

        self.sliders = {
            'equipment_cost': Slider(
                slider_x, start_y, slider_width, slider_height,
                10000, 500000, 100000, "Equipment Cost", " NOK"
            ),
            'quality_prob': Slider(
                slider_x, start_y + spacing, slider_width, slider_height,
                0, 100, 70, "Probability of Quality Increase", "%"
            ),
            'quality_premium': Slider(
                slider_x, start_y + spacing * 2, slider_width, slider_height,
                0, 30, 10, "Quality Premium Increase", "%"
            ),
            'production_increase': Slider(
                slider_x, start_y + spacing * 3, slider_width, slider_height,
                0, 50, 15, "Production Increase", "%"
            ),
            'time_horizon': Slider(
                slider_x, start_y + spacing * 4, slider_width, slider_height,
                1, 15, 5, "Time Horizon", " years"
            )
        }

        # Preset buttons
        self.setup_preset_buttons()

    def setup_preset_buttons(self):
        """Setup preset scenario buttons"""
        button_y = 600
        button_width = 200
        button_height = 40
        spacing = 20
        start_x = 100

        self.preset_buttons = [
            {
                'rect': pygame.Rect(start_x, button_y, button_width, button_height),
                'label': 'Budget Equipment',
                'color': LIGHT_GREEN,
                'values': {'equipment_cost': 50000, 'quality_prob': 40,
                          'quality_premium': 5, 'production_increase': 8, 'time_horizon': 5}
            },
            {
                'rect': pygame.Rect(start_x + button_width + spacing, button_y,
                                   button_width, button_height),
                'label': 'Mid-Range Equipment',
                'color': LIGHT_BLUE,
                'values': {'equipment_cost': 150000, 'quality_prob': 70,
                          'quality_premium': 12, 'production_increase': 18, 'time_horizon': 5}
            },
            {
                'rect': pygame.Rect(start_x + (button_width + spacing) * 2, button_y,
                                   button_width, button_height),
                'label': 'Premium Equipment',
                'color': ORANGE,
                'values': {'equipment_cost': 350000, 'quality_prob': 90,
                          'quality_premium': 20, 'production_increase': 30, 'time_horizon': 5}
            }
        ]

    def apply_preset(self, preset):
        """Apply preset values to sliders"""
        for key, value in preset['values'].items():
            if key in self.sliders:
                self.sliders[key].value = value

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # Handle preset button clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                for preset in self.preset_buttons:
                    if preset['rect'].collidepoint(event.pos):
                        self.apply_preset(preset)

            # Handle slider events
            for slider in self.sliders.values():
                slider.handle_event(event)

    def calculate_results(self):
        """Calculate current results based on slider values"""
        return self.model.calculate_roi(
            equipment_cost=self.sliders['equipment_cost'].value,
            quality_increase_prob=self.sliders['quality_prob'].value / 100,
            quality_premium_increase=self.sliders['quality_premium'].value,
            production_increase_pct=self.sliders['production_increase'].value,
            time_horizon_years=self.sliders['time_horizon'].value
        )

    def draw_results_panel(self, results):
        """Draw the results panel"""
        panel_x = 650
        panel_y = 120
        panel_width = 500
        panel_height = 520

        # Background
        pygame.draw.rect(self.screen, LIGHT_GRAY,
                        (panel_x, panel_y, panel_width, panel_height),
                        border_radius=10)
        pygame.draw.rect(self.screen, DARK_GRAY,
                        (panel_x, panel_y, panel_width, panel_height),
                        3, border_radius=10)

        # Title
        title = HEADER_FONT.render("Financial Analysis", True, BLACK)
        self.screen.blit(title, (panel_x + 20, panel_y + 15))

        # Results
        y_offset = panel_y + 70
        line_spacing = 45

        results_to_show = [
            ("Monthly Revenue (Baseline)", f"{results['baseline_monthly_revenue']:.0f} NOK", BLACK),
            ("Monthly Revenue (New)", f"{results['new_monthly_revenue']:.0f} NOK", BLUE),
            ("Monthly Benefit", f"+{results['monthly_benefit']:.0f} NOK", GREEN if results['monthly_benefit'] > 0 else RED),
            ("Annual Return", f"{results['annual_return']:.0f} NOK", GREEN if results['annual_return'] > 0 else RED),
            ("Total Benefit", f"{results['total_benefit']:.0f} NOK", BLACK),
            ("Equipment Cost", f"-{results['equipment_cost']:.0f} NOK", RED),
            ("Net Benefit", f"{results['net_benefit']:.0f} NOK", GREEN if results['net_benefit'] > 0 else RED),
            ("ROI", f"{results['roi_pct']:.1f}%", GREEN if results['roi_pct'] > 0 else RED),
            ("Payback Period", f"{results['payback_months']:.1f} months" if results['payback_months'] != float('inf') else "Never", BLACK),
        ]

        for i, (label, value, color) in enumerate(results_to_show):
            # Label
            label_surf = LABEL_FONT.render(label + ":", True, BLACK)
            self.screen.blit(label_surf, (panel_x + 30, y_offset + i * line_spacing))

            # Value
            value_surf = VALUE_FONT.render(value, True, color)
            value_rect = value_surf.get_rect(right=panel_x + panel_width - 30)
            value_rect.y = y_offset + i * line_spacing - 2
            self.screen.blit(value_surf, value_rect)

        # Recommendation
        recommendation, rec_color = self.model.get_recommendation(results)
        rec_y = panel_y + panel_height - 70

        pygame.draw.rect(self.screen, rec_color,
                        (panel_x + 20, rec_y, panel_width - 40, 50),
                        border_radius=8)

        rec_text = HEADER_FONT.render(recommendation, True, WHITE)
        rec_rect = rec_text.get_rect(center=(panel_x + panel_width // 2, rec_y + 25))
        self.screen.blit(rec_text, rec_rect)

    def draw_preset_buttons(self):
        """Draw preset scenario buttons"""
        for preset in self.preset_buttons:
            # Button background
            pygame.draw.rect(self.screen, preset['color'], preset['rect'],
                           border_radius=8)
            pygame.draw.rect(self.screen, DARK_GRAY, preset['rect'], 2,
                           border_radius=8)

            # Button label
            label_surf = LABEL_FONT.render(preset['label'], True, BLACK)
            label_rect = label_surf.get_rect(center=preset['rect'].center)
            self.screen.blit(label_surf, label_rect)

        # Section title
        title_surf = LABEL_FONT.render("Preset Scenarios:", True, BLACK)
        self.screen.blit(title_surf, (100, 565))

    def draw(self):
        """Draw everything"""
        self.screen.fill(WHITE)

        # Title
        title = TITLE_FONT.render("Farming Equipment Investment Tool", True, BLACK)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 50))
        self.screen.blit(title, title_rect)

        # Draw sliders
        for slider in self.sliders.values():
            slider.draw(self.screen)

        # Calculate and draw results
        results = self.calculate_results()
        self.draw_results_panel(results)

        # Draw preset buttons
        self.draw_preset_buttons()

        # Instructions
        instructions = SMALL_FONT.render(
            "Drag sliders to adjust parameters or click preset scenarios to compare equipment options",
            True, DARK_GRAY
        )
        self.screen.blit(instructions, (100, 680))

        pygame.display.flip()

    def run(self):
        """Main application loop"""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()


def main():
    """Entry point"""
    app = EquipmentDecisionTool()
    app.run()


if __name__ == "__main__":
    main()
