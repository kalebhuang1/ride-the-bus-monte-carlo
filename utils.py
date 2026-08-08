"""Card helpers and the card-counting policy shared by all three simulations.

House rule: ties lose. Matching the previous rank on Higher or Lower, or landing
exactly on a boundary card on Inside or Outside, is a drink and a restart. The
comparison functions report those as 'equal', `optimal_guess` never returns
'equal', so the caller's `guess != outcome` check charges the drink.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
def red_or_black(card):
    if card.find('♥') != -1 or card.find('♦') != -1:
        return 'red'
    else:
        return 'black'
    
def higher_or_lower(previous_card, new_card):
    """Rank of new_card relative to previous_card: 'higher', 'lower' or 'equal'."""
    rank_order = {rank: i for i, rank in enumerate(ranks)}
    previous_value = rank_order[previous_card[:-1]]
    new_value = rank_order[new_card[:-1]]
    if new_value > previous_value:
        return 'higher'
    elif new_value < previous_value:
        return 'lower'
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


QUESTIONS = ('red_or_black', 'higher_or_lower', 'inside_or_outside', 'suit')

QUESTION_LABELS = {
    'red_or_black': 'Red or Black',
    'higher_or_lower': 'Higher or Lower',
    'inside_or_outside': 'Inside or Outside',
    'suit': 'Suit',
}


def new_counts():
    """Fresh per-question attempt/success tallies for one strategy."""
    return {q: 0 for q in QUESTIONS}, {q: 0 for q in QUESTIONS}


def transition_table(title, tries, wins, mc_mean=None):
    """Per-question transition probabilities estimated from the sampled rounds.

    The game is a Markov chain over the four questions: from question k you
    advance with probability p_k, otherwise you drink and reset to question one.
    Clearing a round has probability P = p1*p2*p3*p4, so expected drinks before a
    win is geometric, (1 - P) / P.

    Note this reproduces the simulated mean exactly rather than approximately,
    and that is an identity, not a validation. Attempts at question k+1 are by
    construction the wins at question k, so P telescopes to wins['suit'] /
    tries['red_or_black'] and (1 - P) / P collapses to total failures over total
    rounds -- which is the sampled mean. It confirms the bookkeeping is
    consistent; it cannot confirm the sampler is right. The per-question
    probabilities are the real content here.
    """
    p = {q: wins[q] / tries[q] if tries[q] else float('nan') for q in QUESTIONS}

    lines = [title, '']
    lines.append(f"{'question':>19}  {'attempts':>10}  {'wins':>10}  {'p(advance)':>11}")
    lines.append(f"{'-' * 19}  {'-' * 10}  {'-' * 10}  {'-' * 11}")
    for q in QUESTIONS:
        lines.append(f"{QUESTION_LABELS[q]:>19}  {tries[q]:>10,}  {wins[q]:>10,}  {p[q]:>10.1%}")

    clear = 1.0
    for q in QUESTIONS:
        clear *= p[q]
    lines.append('')
    lines.append(f"  P(clear all four) = {clear:.5f}   ->   implied E[drinks] = {(1 - clear) / clear:.3f}")
    if mc_mean is not None:
        lines.append(f"  simulated mean drinks = {mc_mean:.3f}   (equal by construction, see docstring)")
    return '\n'.join(lines)


def transition_table_png(path, title, tries, wins, mc_mean=None, rounds=None,
                         accent='#2a78d6'):
    """Render the same numbers as `transition_table` to a PNG for the article.

    A table, not a chart: the point is to read four exact probabilities, not to
    compare a shape. p(advance) carries a single-hue magnitude bar so the four
    questions can still be ranked at a glance, but every number is printed in
    text ink, so nothing here depends on colour to be legible.
    """
    p = {q: wins[q] / tries[q] if tries[q] else float('nan') for q in QUESTIONS}
    clear = 1.0
    for q in QUESTIONS:
        clear *= p[q]

    ink, muted, rule, track = '#2c2b28', '#898781', '#c3c2b7', '#e8e7e1'
    x_q, x_att, x_win, x_bar0, x_bar1, x_pct = 0.015, 0.47, 0.605, 0.65, 0.86, 0.985

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Keep the title short and push the run details to a subtitle -- a long title
    # overruns the table rules and the three strategies stop looking like a set.
    ax.text(x_q, 0.94, title, fontsize=14, fontweight='bold', color=ink)
    details = []
    if rounds is not None:
        details.append(f'{rounds:,} simulated rounds')
    if mc_mean is not None:
        details.append(f'{mc_mean:.3f} drinks per completed round')
    if details:
        ax.text(x_q, 0.855, '   ·   '.join(details), fontsize=10.5, color=muted)

    for x, label, align in ((x_q, 'question', 'left'), (x_att, 'attempts', 'right'),
                            (x_win, 'wins', 'right'), (x_bar0, 'p(advance)', 'left')):
        ax.text(x, 0.70, label, fontsize=9.5, color=muted, ha=align)
    ax.plot([0, 1], [0.665, 0.665], color=rule, linewidth=1)

    y = 0.555
    for q in QUESTIONS:
        ax.text(x_q, y, QUESTION_LABELS[q], fontsize=11.5, color=ink)
        ax.text(x_att, y, f'{tries[q]:,}', fontsize=11.5, color=muted, ha='right')
        ax.text(x_win, y, f'{wins[q]:,}', fontsize=11.5, color=muted, ha='right')
        # Track spans the full 0-100% width so the bars are comparable across files.
        ax.add_patch(plt.Rectangle((x_bar0, y - 0.005), x_bar1 - x_bar0, 0.045,
                                   facecolor=track, edgecolor='none'))
        ax.add_patch(plt.Rectangle((x_bar0, y - 0.005), (x_bar1 - x_bar0) * p[q], 0.045,
                                   facecolor=accent, edgecolor='none'))
        ax.text(x_pct, y, f'{p[q]:.1%}', fontsize=11.5, color=ink,
                ha='right', fontweight='bold')
        y -= 0.125

    ax.plot([0, 1], [0.135, 0.135], color=rule, linewidth=1)
    ax.text(x_q, 0.045,
            f'P(clear all four) = {clear:.5f}      implied E[drinks] = (1 - P) / P = '
            f'{(1 - clear) / clear:.3f}',
            fontsize=10, color=muted)

    # Fixed margins rather than tight bbox, so all three tables render identical
    # in size and can be stacked in the article without rescaling.
    fig.subplots_adjust(left=0.045, right=0.955, top=0.95, bottom=0.05)
    fig.savefig(path, dpi=200, facecolor='white')
    return fig


def reshuffle(deck, in_play=()):
    """Rebuild the draw pile once it runs dry.

    Everything goes back in except `in_play` — the cards currently face-up on the
    table for the question being asked. Those stay out, so a mid-round reshuffle
    can never redeal a card the player is already looking at.
    """
    return list(np.random.permutation([c for c in deck if c not in in_play]))


def optimal_guess(deck, question_type, card1 = None, card2 = None):
    if question_type == 'red_or_black':
        red_count = sum(1 for card in deck if red_or_black(card) == 'red')
        black_count = sum(1 for card in deck if red_or_black(card) == 'black')
        if red_count > black_count:
            return 'red'
        elif red_count == black_count:
            return np.random.choice(['red', 'black'])
        else:
            return 'black'
    elif question_type == 'higher_or_lower':
        # Ties lose whichever way you call it, so cards matching card1's rank are
        # dead weight and stay out of both counts.
        higher_count = sum(1 for card in deck if higher_or_lower(card1, card) == 'higher')
        lower_count = sum(1 for card in deck if higher_or_lower(card1, card) == 'lower')
        if higher_count > lower_count:
            return 'higher'
        elif higher_count == lower_count:
            return np.random.choice(['higher', 'lower'])
        else:
            return 'lower'
    elif question_type == 'inside_or_outside':
        # Same as above: the boundary ranks lose either way, so they stay out of
        # both counts rather than being folded into 'outside'.
        inside_count = sum(1 for card in deck if inside_or_outside(card1, card2, card) == 'inside')
        outside_count = sum(1 for card in deck if inside_or_outside(card1, card2, card) == 'outside')
        if inside_count > outside_count:
            return 'inside'
        elif inside_count == outside_count:
            return np.random.choice(['inside', 'outside'])
        else:
            return 'outside'
    elif question_type == 'suit':
        suit_counts = {s: sum(1 for card in deck if suit(card) == s) for s in ['♠', '♥', '♦', '♣']}
        max_count = max(suit_counts.values())
        best = [s for s, count in suit_counts.items() if count == max_count]
        return np.random.choice(best)
    else:
        raise ValueError(f"unknown question_type: {question_type!r}")
