import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import red_or_black, higher_or_lower, inside_or_outside, suit, optimal_guess

np.random.seed(42)  # reproducibility

N_TRIALS = 50000

suits = ['♠', '♥', '♦', '♣']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

deck = [f"{rank}{suit}" for suit in suits for rank in ranks]

drinks_obs = []

for i in range(N_TRIALS):
    finished = False
    shuffled_deck = list(np.random.permutation(deck))
    drinks_round = 0
    while not finished:
        if len(shuffled_deck) == 0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_red_black = optimal_guess(shuffled_deck, 'red_or_black')
        card = shuffled_deck.pop()
        red_or_black_guess = red_or_black(card)
        if guess_red_black != red_or_black_guess:
            drinks_round += 1
            continue
        if len(shuffled_deck) == 0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_higher_lower = optimal_guess(shuffled_deck, 'higher_or_lower', card, None)
        card2 = shuffled_deck.pop()
        higher_or_lower_guess = higher_or_lower(card, card2)
        if guess_higher_lower != higher_or_lower_guess:
            drinks_round += 1
            continue
        if len(shuffled_deck) == 0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_inside_outside = optimal_guess(shuffled_deck, 'inside_or_outside', card, card2)
        card3 = shuffled_deck.pop()
        inside_or_outside_guess = inside_or_outside(card, card2, card3)
        if guess_inside_outside != inside_or_outside_guess:
            drinks_round += 1
            continue
        if len(shuffled_deck) == 0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_suit = optimal_guess(shuffled_deck, 'suit')
        card4 = shuffled_deck.pop()
        suit_guess = suit(card4)
        if guess_suit != suit_guess:
            drinks_round += 1
            continue
        finished = True
        drinks_obs.append(drinks_round)
print(f"Average drinks per round: {np.mean(drinks_obs)}")

# --- Monte Carlo convergence: running mean of drinks per round ---
drinks = np.asarray(drinks_obs, dtype=float)
n = np.arange(1, drinks.size + 1)

cum_mean = np.cumsum(drinks) / n

# Running sample variance -> standard error of the running mean (s_n / sqrt(n)).
cum_sq = np.cumsum(drinks ** 2) / n
cum_var = np.maximum(cum_sq - cum_mean ** 2, 0.0) * n / np.maximum(n - 1, 1)
se = np.sqrt(cum_var / n)

final = cum_mean[-1]
final_sd = np.sqrt(cum_var[-1])   # spread of a single round
final_se = se[-1]                 # uncertainty of the estimate

print(f"Standard deviation: {final_sd:.3f}")
print(f"Standard error:     {final_se:.3f}")

fig, (ax, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))

ax.fill_between(n, cum_mean - 1.96 * se, cum_mean + 1.96 * se,
                color='#2a78d6', alpha=0.15, linewidth=0,
                label=r'95% Monte Carlo error')
ax.axhline(final, color='#898781', linewidth=1.5, linestyle='--',
           label=f'Final estimate = {final:.3f}')
ax.plot(n, cum_mean, color='#2a78d6', linewidth=2, label='Running mean')

# Text-only legend entries for the summary statistics.
ax.plot([], [], ' ', label=f'Standard deviation  $s$ = {final_sd:.2f}')
ax.plot([], [], ' ', label=f'Standard error  $s/\\sqrt{{n}}$ = {final_se:.3f}')

ax.set_xlabel('Simulated rounds')
ax.set_ylabel('Mean drinks per round')
ax.set_title('Convergence of the Monte Carlo estimate', fontsize=13, fontweight='bold')

ax.set_xlim(1, drinks.size)
tail = cum_mean[min(50, drinks.size - 1):]
pad = max(0.35, 0.6 * (tail.max() - tail.min()))
ax.set_ylim(tail.min() - pad, tail.max() + pad)

ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_color('#c3c2b7')
ax.tick_params(colors='#898781')
ax.legend(frameon=False, loc='upper right')

# --- Cumulative distribution: share of rounds costing at most x drinks ---
x_max = np.ceil(np.percentile(drinks, 99) / 10) * 10
ordered = np.sort(drinks)
share = np.arange(1, drinks.size + 1) / drinks.size

ax_cdf.step(ordered, share, where='post', color='#2a78d6', linewidth=2)

# Read the median and 90th percentile straight off the curve
for quantile, name in ((0.5, 'median'), (0.9, '90th pct')):
    value = np.percentile(drinks, quantile * 100)
    ax_cdf.plot([0, value], [quantile, quantile], color='#c3c2b7', linewidth=1)
    ax_cdf.plot([value, value], [0, quantile], color='#c3c2b7', linewidth=1)
    ax_cdf.annotate(f'{name}: {value:.0f} drinks', xy=(value, quantile),
                    xytext=(8, -14), textcoords='offset points',
                    fontsize=10, color='#52514e')

ax_cdf.set_xlim(0, x_max)
ax_cdf.set_ylim(0, 1)
ax_cdf.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax_cdf.set_xlabel('Drinks in a round')
ax_cdf.set_ylabel('Share of rounds costing this much or less')
ax_cdf.set_title('Cumulative distribution', fontsize=13, fontweight='bold')

ax_cdf.grid(axis='y', color='#e1e0d9', linewidth=0.8)
ax_cdf.set_axisbelow(True)
for side in ('top', 'right'):
    ax_cdf.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax_cdf.spines[side].set_color('#c3c2b7')
ax_cdf.tick_params(colors='#898781')

fig.suptitle(f'Ride the Bus — optimal play ({drinks.size:,} simulated rounds)',
             fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig('convergence_optimal_play.png', dpi=150)
plt.show()
