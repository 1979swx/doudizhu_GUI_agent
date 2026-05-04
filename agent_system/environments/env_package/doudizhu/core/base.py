# Copyright (c) 2019 DATA Lab at Texas A&M University
# Adapted from RLCard for the verl-agent Dou Dizhu environment.

class Card:
    """A single playing card."""

    valid_suit = ["S", "H", "D", "C", "BJ", "RJ"]
    valid_rank = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]

    def __init__(self, suit, rank):
        self.suit = suit
        self.rank = rank

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.rank == other.rank and self.suit == other.suit
        return NotImplemented

    def __hash__(self):
        suit_index = Card.valid_suit.index(self.suit)
        rank_index = Card.valid_rank.index(self.rank)
        return rank_index + 100 * suit_index

    def __str__(self):
        return self.rank + self.suit

    def get_index(self):
        return self.suit + self.rank


def init_54_deck():
    suit_list = ["S", "H", "D", "C"]
    rank_list = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K"]
    deck = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    deck.append(Card("BJ", ""))
    deck.append(Card("RJ", ""))
    return deck
