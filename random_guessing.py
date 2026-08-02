import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # reproducibility

N_TRIALS = 5000
VERBOSE = False 

suits = ['♠', '♥', '♦', '♣']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

deck = [f"{rank}{suit}" for suit in suits for rank in ranks]

print(deck)

drinks_obs = []

def red_or_black(card):
    if card.find('♥') != -1 or card.find('♦') != -1:
        return 'red'
    else:
        return 'black'
    
def higher_or_lower(card1, card2):
    value1 = card1[:-1]
    value2 = card2[:-1]
    rank_order = {rank: i for i, rank in enumerate(ranks)}
    if rank_order[value1] < rank_order[value2]:
        return 'lower'
    elif rank_order[value1] > rank_order[value2]:
        return 'higher'
    else:
        return 'equal'
def inside_or_outside(card1, card2, card3):
    value1 = card1[:-1]
    value2 = card2[:-1]
    value3 = card3[:-1]
    rank_order = {rank: i for i, rank in enumerate(ranks)}
    min_value = min(rank_order[value1], rank_order[value2])
    max_value = max(rank_order[value1], rank_order[value2])
    if rank_order[value3] > min_value and rank_order[value3] < max_value:
        return 'inside'
    elif rank_order[value3] < min_value or rank_order[value3] > max_value:
        return 'outside'
    else:
        return 'equal'
def suit(card):
    return card[-1]
 
for i in range(N_TRIALS):
    finished = False
    shuffled_deck = list(np.random.permutation(deck))
    drinks_round = 0
    while not finished:
        if len(shuffled_deck) ==0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_red_black = np.random.choice(['red', 'black'])
        card = shuffled_deck.pop()
        red_or_black_guess = red_or_black(card)
        if guess_red_black != red_or_black_guess:
            drinks_round += 1
            if VERBOSE:
                print('drink!')
            continue
        if len(shuffled_deck) ==0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_higher_lower = np.random.choice(['higher', 'lower'])
        card2 = shuffled_deck.pop()
        higher_or_lower_guess = higher_or_lower(card, card2)
        if guess_higher_lower != higher_or_lower_guess:
            drinks_round += 1
            if VERBOSE:
                print('drink!')
            continue
        if len(shuffled_deck) ==0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_inside_outside = np.random.choice(['inside', 'outside'])
        card3 = shuffled_deck.pop()
        inside_or_outside_guess = inside_or_outside(card, card2, card3)
        if guess_inside_outside != inside_or_outside_guess:
            drinks_round += 1
            if VERBOSE:
                print('drink!')
            continue
        if len(shuffled_deck) ==0:
            shuffled_deck = list(np.random.permutation(deck))
        guess_suit = np.random.choice(['♠', '♥', '♦', '♣'])
        card4 = shuffled_deck.pop()
        suit_guess = suit(card4)
        if guess_suit != suit_guess:
            drinks_round += 1
            if VERBOSE:
                print('drink!')
            continue
        finished = True
        drinks_obs.append(drinks_round)
print(f"Average drinks: {np.mean(drinks_obs)}")

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

fig, ax = plt.subplots(figsize=(9, 5))

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
ax.set_title('Convergence of Monte Carlo estimate of drinks per round (Random Guessing)', fontsize=14, fontweight='bold')

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

fig.tight_layout()
fig.savefig('convergence_random_guessing.png', dpi=150)
plt.show()

    