from manim import *
import numpy as np


class Credits(Scene):
    def construct(self):
        line1 = Text("Animation by Caleb Hayton", font_size=48)
        line2 = Text("CWH_Physics", font_size=36, color=BLUE)
        credit = VGroup(line1, line2).arrange(DOWN, buff=0.4)

        self.play(FadeIn(credit, shift=UP * 0.3), run_time=1)
        self.play(Flash(line2, color=BLUE, line_length=0.4, flash_radius=1.6))
        self.wait(1.5)
        self.play(FadeOut(credit), run_time=1)


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def weird_pdf_raw(x):
    """An arbitrary, deliberately non-Gaussian shape on [0, 1]:
    a flat block, a rising ramp and a sharp spike."""
    x = np.asarray(x, dtype=float)
    block = _sigmoid((x - 0.05) / 0.008) * _sigmoid((0.30 - x) / 0.008)
    ramp = np.clip((x - 0.45) / 0.15, 0, 1) * _sigmoid((0.60 - x) / 0.005)
    spike = np.exp(-0.5 * ((x - 0.82) / 0.025) ** 2)
    return 1.0 * block + 1.2 * ramp + 1.6 * spike


class CLTDemo(Scene):
    SAMPLE_SIZE = 10  # points averaged per sample
    NUM_BINS = 40

    def construct(self):
        rng = np.random.default_rng(7)
        n = self.SAMPLE_SIZE

        # --- The distribution: normalise on a grid, build an inverse CDF ---
        grid = np.linspace(0, 1, 4001)
        raw = weird_pdf_raw(grid)
        norm = np.trapezoid(raw, grid)
        pdf_vals = raw / norm
        cdf = np.cumsum(pdf_vals)
        cdf /= cdf[-1]
        mu = np.trapezoid(grid * pdf_vals, grid)
        sigma = np.sqrt(np.trapezoid((grid - mu) ** 2 * pdf_vals, grid))

        def pdf(x):
            return float(np.interp(x, grid, pdf_vals))

        def draw(k):
            return np.interp(rng.random(k), cdf, grid)

        # --- Layout ---
        left_ax = Axes(
            x_range=[0, 1, 0.1], y_range=[0, pdf_vals.max() * 1.15, 1],
            x_length=5.6, y_length=3.6, tips=False,
        ).to_edge(LEFT, buff=0.6).shift(DOWN * 0.5)
        right_ax = Axes(
            x_range=[0, 1, 0.1], y_range=[0, 1.15, 1],
            x_length=5.6, y_length=3.6, tips=False,
            y_axis_config={"include_ticks": False},
        ).to_edge(RIGHT, buff=0.6).shift(DOWN * 0.5)

        left_title = Text("A weird distribution", font_size=30).next_to(left_ax, UP, buff=0.35)
        right_title = Text("Means of the samples", font_size=30).next_to(right_ax, UP, buff=0.35)

        curve = left_ax.plot(pdf, x_range=[0, 1, 0.002], color=BLUE, use_smoothing=False)
        area = left_ax.get_area(curve, x_range=[0, 1], color=BLUE, opacity=0.25)

        info = Text(f"Each sample: average of n = {n} random points", font_size=26)
        info.to_edge(UP, buff=0.3)

        self.play(Create(left_ax), Write(left_title), run_time=1)
        self.play(Create(curve), FadeIn(area), run_time=2)
        self.wait(0.5)
        self.play(Create(right_ax), Write(right_title), FadeIn(info), run_time=1)

        # --- Histogram of the means ---
        edges = np.linspace(0, 1, self.NUM_BINS + 1)
        bin_w = edges[1] - edges[0]
        counts = np.zeros(self.NUM_BINS)
        state = {"total": 0}

        def make_bars():
            peak = max(counts.max(), 1)
            bars = VGroup()
            for i, c in enumerate(counts):
                h = c / peak
                rect = Rectangle(
                    width=right_ax.x_axis.unit_size * bin_w,
                    height=max(right_ax.y_axis.unit_size * h, 1e-3),
                    stroke_width=1 if c else 0,
                    stroke_color=WHITE,
                    fill_color=GREEN,
                    fill_opacity=0.75 if c else 0,
                )
                rect.move_to(right_ax.c2p(edges[i] + bin_w / 2, 0), aligned_edge=DOWN)
                bars.add(rect)
            return bars

        def make_counter():
            return Text(f"Samples: {state['total']}", font_size=26).next_to(
                right_ax, DOWN, buff=0.3
            )

        def add_means(means):
            idx = np.clip(np.searchsorted(edges, means, side="right") - 1, 0, self.NUM_BINS - 1)
            np.add.at(counts, idx, 1)
            state["total"] += len(means)

        def bar_top(m):
            i = min(int(m / bin_w), self.NUM_BINS - 1)
            peak = max(counts.max(), 1)
            return right_ax.c2p(m, counts[i] / peak)

        bars = make_bars()
        counter = make_counter()
        self.add(bars)
        self.play(FadeIn(counter))

        # --- Phase 1 & 2: individual samples, slow then fast ---
        def one_round(speed):
            xs = draw(n)
            m = xs.mean()
            jitter = rng.uniform(0.05, 0.25, n)
            dots = VGroup(*[
                Dot(left_ax.c2p(x, 0) + UP * j, radius=0.06, color=ORANGE)
                for x, j in zip(xs, jitter)
            ])
            mean_line = DashedLine(
                left_ax.c2p(m, 0), left_ax.c2p(m, left_ax.y_range[1]), color=YELLOW
            )
            mean_dot = Dot(left_ax.c2p(m, 0), radius=0.09, color=YELLOW)

            self.play(
                LaggedStart(*[FadeIn(d, shift=DOWN * 0.4) for d in dots], lag_ratio=0.15),
                run_time=1.2 * speed,
            )
            self.play(Create(mean_line), FadeIn(mean_dot), run_time=0.5 * speed)

            add_means(np.array([m]))
            new_bars = make_bars()
            self.play(
                mean_dot.animate.move_to(bar_top(m)),
                FadeOut(dots),
                FadeOut(mean_line),
                run_time=0.8 * speed,
            )
            self.play(
                Transform(bars, new_bars),
                Transform(counter, make_counter()),
                FadeOut(mean_dot),
                run_time=0.3 * speed,
            )

        for _ in range(3):
            one_round(speed=1.0)
        for _ in range(12):
            one_round(speed=0.35)

        # --- Phase 3: bulk sampling ---
        for target in [30, 60, 120, 250, 500, 1000, 2000, 4000, 8000]:
            batch = draw((target - state["total"]) * n).reshape(-1, n).mean(axis=1)
            add_means(batch)
            self.play(
                Transform(bars, make_bars()),
                Transform(counter, make_counter()),
                run_time=0.6,
            )

        # --- Phase 4: overlay the predicted normal curve ---
        sd_mean = sigma / np.sqrt(n)
        scale = state["total"] * bin_w / counts.max()

        def normal(x):
            return scale * np.exp(-0.5 * ((x - mu) / sd_mean) ** 2) / (sd_mean * np.sqrt(2 * PI))

        gauss = right_ax.plot(normal, x_range=[0, 1, 0.002], color=YELLOW, stroke_width=4)
        gauss_label = Text("Normal (bell) curve", font_size=26, color=YELLOW)
        gauss_label.next_to(right_ax.c2p(mu, 1.05), RIGHT, buff=0.3)

        self.play(Create(gauss), run_time=2)
        self.play(FadeIn(gauss_label, shift=LEFT * 0.2))
        self.wait(3)
