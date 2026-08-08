import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import (red_or_black, higher_or_lower, inside_or_outside, suit,
                   optimal_guess, reshuffle, new_counts, transition_table,
                   transition_table_png)

np.random.seed(42)

N_TRIALS = 50000

suits = ['♠', '♥', '♦', '♣']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

deck = [f"{rank}{suit}" for suit in suits for rank in ranks]

drinks_obs = []
tries, wins = new_counts()

def add_7_card(seen, card):
    if len(seen) < 7:
        seen.append(card)
        return seen
    else:
        seen.pop(0)
        seen.append(card)
        return seen

def unseen_pool(seen):
    return [c for c in deck if c not in seen]

# game logic
for i in range(N_TRIALS):
    finished = False
    shuffled_deck = list(np.random.permutation(deck))
    drinks_round = 0
    seen = []
    while not finished:
        if len(shuffled_deck) == 0:
            shuffled_deck = reshuffle(deck)
            seen = []
        tries['red_or_black'] += 1
        guess_red_black = optimal_guess(unseen_pool(seen), 'red_or_black')
        card = shuffled_deck.pop()
        seen = add_7_card(seen, card)
        red_or_black_guess = red_or_black(card)
        if guess_red_black != red_or_black_guess:
            drinks_round += 1
            continue
        wins['red_or_black'] += 1
        if len(shuffled_deck) == 0:
            shuffled_deck = reshuffle(deck, [card])
            seen = [card]
        tries['higher_or_lower'] += 1
        guess_higher_lower = optimal_guess(unseen_pool(seen), 'higher_or_lower', card, None)
        card2 = shuffled_deck.pop()
        seen = add_7_card(seen, card2)
        higher_or_lower_guess = higher_or_lower(card, card2)
        if guess_higher_lower != higher_or_lower_guess:
            drinks_round += 1
            continue
        wins['higher_or_lower'] += 1
        if len(shuffled_deck) == 0:
            shuffled_deck = reshuffle(deck, [card, card2])
            seen = [card, card2]
        tries['inside_or_outside'] += 1
        guess_inside_outside = optimal_guess(unseen_pool(seen), 'inside_or_outside', card, card2)
        card3 = shuffled_deck.pop()
        seen = add_7_card(seen, card3)
        inside_or_outside_guess = inside_or_outside(card, card2, card3)
        if guess_inside_outside != inside_or_outside_guess:
            drinks_round += 1
            continue
        wins['inside_or_outside'] += 1
        if len(shuffled_deck) == 0:
            shuffled_deck = reshuffle(deck, [card, card2, card3])
            seen = [card, card2, card3]
        tries['suit'] += 1
        guess_suit = optimal_guess(unseen_pool(seen), 'suit')
        card4 = shuffled_deck.pop()
        seen = add_7_card(seen, card4)
        suit_guess = suit(card4)
        if guess_suit != suit_guess:
            drinks_round += 1
            continue
        wins['suit'] += 1
        finished = True
        drinks_obs.append(drinks_round)
# table
print(f"Average drinks per round: {np.mean(drinks_obs)}")
print()
print(transition_table(
    f"Last-7 memory — transition probabilities ({len(drinks_obs):,} rounds)",
    tries, wins, np.mean(drinks_obs)))
print()

transition_table_png('transitions_7_card.png', 'Last-7 memory',
                     tries, wins, np.mean(drinks_obs), len(drinks_obs))

# mc plot
drinks = np.asarray(drinks_obs, dtype=float)
n = np.arange(1, drinks.size + 1)

cum_mean = np.cumsum(drinks) / n

cum_sq = np.cumsum(drinks ** 2) / n
cum_var = np.maximum(cum_sq - cum_mean ** 2, 0.0) * n / np.maximum(n - 1, 1)
se = np.sqrt(cum_var / n)

final = cum_mean[-1]
final_sd = np.sqrt(cum_var[-1])
final_se = se[-1]

print(f"Standard deviation: {final_sd:.3f}")
print(f"Standard error:     {final_se:.3f}")

fig, (ax, ax_cdf) = plt.subplots(1, 2, figsize=(14, 5))

ax.fill_between(n, cum_mean - 1.96 * se, cum_mean + 1.96 * se,
                color='#2a78d6', alpha=0.15, linewidth=0,
                label=r'95% Monte Carlo error')
ax.axhline(final, color='#898781', linewidth=1.5, linestyle='--',
           label=f'Final estimate = {final:.3f}')
ax.plot(n, cum_mean, color='#2a78d6', linewidth=2, label='Running mean')

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

# cdf plot
x_max = np.ceil(np.percentile(drinks, 99) / 10) * 10
ordered = np.sort(drinks)
share = np.arange(1, drinks.size + 1) / drinks.size

ax_cdf.step(ordered, share, where='post', color='#2a78d6', linewidth=2)

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

fig.suptitle(f'Ride the Bus — 7-card memory ({drinks.size:,} simulated rounds)',
             fontsize=15, fontweight='bold')
fig.tight_layout()
fig.savefig('convergence_7_card.png', dpi=150)


#histogram
bin_width = 1
edges = np.arange(0, x_max + bin_width, bin_width)
counts, _ = np.histogram(drinks, bins=edges)

fig_hist, ax_hist = plt.subplots(figsize=(9, 5))

ax_hist.bar(edges[:-1], counts / drinks.size, width=bin_width * 0.85,
            align='edge', color='#2a78d6', linewidth=0,
            label='Simulated rounds')

p_clear = 1 / (1 + drinks.mean())
pmf = p_clear * (1 - p_clear) ** np.arange(0, int(x_max))
model = pmf.reshape(-1, bin_width).sum(axis=1)
ax_hist.plot(edges[:-1] + bin_width / 2, model, color='#52514e',
             linewidth=2, linestyle='--',
             label=f'Geometric($p$ = {p_clear:.4f})')

mean_v = drinks.mean()
median_v = np.median(drinks)
top = max(counts.max() / drinks.size, model.max())
for value, name, height in ((median_v, 'median', 0.93), (mean_v, 'mean', 0.78)):
    ax_hist.axvline(value, color='#c3c2b7', linewidth=1.2)
    ax_hist.annotate(f'{name}: {value:.1f}', xy=(value, top * height),
                     xytext=(6, 0), textcoords='offset points',
                     fontsize=10, color='#52514e')

ax_hist.set_xlim(0, x_max)
ax_hist.set_ylim(0, top * 1.05)
ax_hist.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax_hist.set_xlabel('Drinks in a round')
ax_hist.set_ylabel('Share of rounds')
ax_hist.set_title(f'Ride the Bus — 7-card memory ({drinks.size:,} simulated rounds)',
                  fontsize=13, fontweight='bold')

ax_hist.grid(axis='y', color='#e1e0d9', linewidth=0.8)
ax_hist.set_axisbelow(True)
for side in ('top', 'right'):
    ax_hist.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax_hist.spines[side].set_color('#c3c2b7')
ax_hist.tick_params(colors='#898781')
ax_hist.legend(frameon=False, loc='upper right')

fig_hist.tight_layout()
fig_hist.savefig('histogram_7_card.png', dpi=150)

plt.show()