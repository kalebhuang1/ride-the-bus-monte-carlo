import pandas as pd
import numpy as np

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
        higher_count = sum(1 for card in deck if higher_or_lower(card1, card) == 'higher')
        lower_count = sum(1 for card in deck if higher_or_lower(card1, card) == 'lower')
        if higher_count > lower_count:
            return 'higher'
        elif higher_count == lower_count:
            return np.random.choice(['higher', 'lower'])
        else:
            return 'lower'
    elif question_type == 'inside_or_outside':
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
