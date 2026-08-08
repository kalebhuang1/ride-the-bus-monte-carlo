import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils import (red_or_black, higher_or_lower, inside_or_outside, suit,
                   optimal_guess, reshuffle, new_counts, QUESTIONS, QUESTION_LABELS)

N_TRIALS = 50000

suits = ['♠', '♥', '♦', '♣']
ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

deck = [f"{rank}{suit}" for suit in suits for rank in ranks]

STRATEGIES = (
    ('random',  'Random',        '#4a3aa7'),
    ('last7',   'Last-7 memory', '#2a78d6'),
    ('optimal', 'Optimal',       '#1baf7a'),
)

COIN = {'red_or_black': ['red', 'black'],
        'higher_or_lower': ['higher', 'lower'],
        'inside_or_outside': ['inside', 'outside'],
        'suit': suits}

PUBLISHED = {'random': 37.80718, 'last7': 13.66434, 'optimal': 10.73798}

# game logic
def add_7_card(seen, card):
    if len(seen) < 7:
        seen.append(card)
        return seen
    else:
        seen.pop(0)
        seen.append(card)
        return seen

def run(strategy):
    np.random.seed(42)
    tries, wins = new_counts()
    drinks_obs = []
    for _ in range(N_TRIALS):
        shuffled_deck = list(np.random.permutation(deck))
        drinks_round = 0
        seen = []
        finished = False
        while not finished:
            table = []
            cleared = True
            for question in QUESTIONS:
                if len(shuffled_deck) == 0:
                    shuffled_deck = reshuffle(deck, table)
                    seen = list(table)
                tries[question] += 1
                if strategy == 'random':
                    guess = np.random.choice(COIN[question])
                else:
                    pool = (shuffled_deck if strategy == 'optimal'
                            else [c for c in deck if c not in seen])
                    guess = optimal_guess(pool, question,
                                          table[0] if table else None,
                                          table[1] if len(table) > 1 else None)
                card = shuffled_deck.pop()
                seen = add_7_card(seen, card)
                if question == 'red_or_black':
                    ok = guess == red_or_black(card)
                elif question == 'higher_or_lower':
                    ok = guess == higher_or_lower(table[0], card)
                elif question == 'inside_or_outside':
                    ok = guess == inside_or_outside(table[0], table[1], card)
                else:
                    ok = guess == suit(card)
                table = table + [card]
                if not ok:
                    drinks_round += 1
                    cleared = False
                    break
                wins[question] += 1
            if cleared:
                finished = True
                drinks_obs.append(drinks_round)
    return np.asarray(drinks_obs, dtype=float), tries, wins

results = {}
for key, label, _ in STRATEGIES:
    drinks, tries, wins = run(key)
    results[key] = (drinks, tries, wins)
    drift = drinks.mean() - PUBLISHED[key]
    flag = '' if abs(drift) < 0.05 else f'   <-- DRIFT from {PUBLISHED[key]:.3f}'
    print(f'{label:>16}: {drinks.mean():7.3f} drinks   median {np.median(drinks):5.1f}{flag}')

# comparison plot
fig, (ax, ax_q) = plt.subplots(1, 2, figsize=(14, 5.5))

x_max = np.ceil(np.percentile(results['random'][0], 90) / 10) * 10

handles = []
for key, label, color in STRATEGIES:
    drinks = results[key][0]
    ordered = np.sort(drinks)
    share = np.arange(1, drinks.size + 1) / drinks.size
    line, = ax.step(ordered, share, where='post', color=color, linewidth=2.2,
                    label=f'{label} — {drinks.mean():.1f} drinks avg')
    handles.append(line)

ax.set_xlim(0, x_max)
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_xlabel('Drinks in a round')
ax.set_ylabel('Share of rounds costing this much or less')
ax.set_title('What a round costs', fontsize=13, fontweight='bold')
ax.grid(axis='y', color='#e1e0d9', linewidth=0.8)
ax.set_axisbelow(True)

height = 0.24
for i, question in enumerate(QUESTIONS):
    for j, (key, label, color) in enumerate(STRATEGIES):
        tries, wins = results[key][1], results[key][2]
        p = wins[question] / tries[question]
        y = i + (j - 1) * (height + 0.03)
        ax_q.barh(y, p, height=height, color=color, linewidth=0)
        ax_q.text(p + 0.008, y, f'{p:.1%}', va='center', fontsize=9.5,
                  color='#2c2b28', fontweight='bold')

ax_q.set_yticks(range(len(QUESTIONS)))
ax_q.set_yticklabels([QUESTION_LABELS[q] for q in QUESTIONS])
ax_q.invert_yaxis()
ax_q.set_xlim(0, 0.85)
ax_q.xaxis.set_major_formatter(PercentFormatter(xmax=1))
ax_q.set_xlabel('Chance of clearing the question')
ax_q.set_title('Where the strategies separate', fontsize=13, fontweight='bold')
ax_q.grid(axis='x', color='#e1e0d9', linewidth=0.8)
ax_q.set_axisbelow(True)

for a in (ax, ax_q):
    for side in ('top', 'right'):
        a.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        a.spines[side].set_color('#c3c2b7')
    a.tick_params(colors='#898781')

fig.suptitle(f'Ride the Bus — three strategies, {N_TRIALS:,} rounds each',
             fontsize=15, fontweight='bold', y=0.985)
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.945), fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig('comparison.png', dpi=150)
plt.show()
