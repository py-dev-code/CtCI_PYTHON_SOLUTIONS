    
'''Two Player Blackjack - Melissa Morel'''

import random
# initializing constants
CARDS = ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K')
RANKS = {'A': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
         '9': 9, '10': 10, 'J': 10, 'Q': 10, 'K': 10}


class Card:
    '''Card consists of one element in RANKS.
    This class is capable of returning the
    value of the rank and also returning a
    printable string representation of a card.'''

    def __init__(self, rank):
        '''rank refers to an element in RANKS.
        If the value exists in RANKS, initialize it.
        Else, print "Invalid card: 'rank'"'''
        if(rank in RANKS):
            self.rank = rank
        else:
            self.rank = None
            print("Invalid card: ", rank)

    # Returns the value of this card's rank
    def get_rank(self):
        return self.rank

    # Returns printable string representation of the card
    def __str__(self):
        return str(self.rank)


class Hand:
    '''Hand consists of a list of cards.
    This class is capable of adding
    a card, returning the total value of the
    hand, printing face up cards, showing the
    player their full hand if they consent,
    checking if the player has a bust, and
    returning a printable representation
    of a hand.'''

    def __init__(self):
        # hand refers to a list of cards
        self.hand = []

    def add_card(self, card):
        # Adds a card to the list
        self.hand.append(card)

    def get_value(self):
        # Returns the value of a hand
        value = 0
        ace = 0
        for card in self.hand:
            value += RANKS[card.get_rank()]
            if str(card.get_rank()) == 'A':
                ace += 1

            for _ in range(ace):
                if value <= 11:
                    '''If the value of the hand is <= 11, we want an Ace to be worth 11 points.
                    Since 1 has already been added because 'A' : 1 in VALUES,
                    we only need to add 10 additional points'''
                    value += 10

        return value

    def print_face_up_cards(self):
        '''Return cards visible to both players
        (everything except the first card)'''
        for i in range(1, (len(self.hand))):
            print(self.hand[i], end=' ')
        print()

    def viewHand(self):
        '''Asks the player if they want to view their hand and
        proceeds accordingly'''
        global turn
        view = input("\n" + turn + ": Do you want to view your full hand? Type"
                     "\'yes\' to view hand, type 'no' to continue.\n")

        # User input check
        while view.lower() != 'yes' and view.lower() != 'no':
            view = input(turn + ": Please type 'yes' or 'no' to continue.\n")

        if view.lower() == 'yes':
            print("\n" + turn + "'s Hand: ", self, sep="")
            print(turn + "'s Hand total: ", self.get_value(), sep="")
        print()

    # Check if the player's hand is over 21 (bust)
    def bust(self):
        global turn, playerBust, player1, player2

        # If the player has a bust, proceed according to player
        if self.get_value() > 21:
            playerBust = True
            if turn == 'Player 1':
                print(turn + ": you have gone over 21. You're busted! Player 2\
                      wins!")
                # Print Player 1's hand and total
                print("Player 1's Hand: ", self)
                print("Player 1's Hand total: ", self.get_value())
                # Print Player 2's hand and total
                print("Player 2's Hand: ", player2)
                print("Player 2's Hand total: ", player2.get_value())
            if turn == 'Player 2':
                print(turn + ": you have gone over 21. You're busted! Player 1\
                      wins!")
                # Print Player 1's hand and total
                print("Player 1's Hand: ", player1)
                print("Player 1's Hand total: ", player1.get_value())
                # Print Player 2's hand and total
                print("Player 2's Hand: ", self)
                print("Player 2's Hand total: ", self.get_value())

    # Returns printable string representation of the hand
    def __str__(self):
        cards = ''
        for card in self.hand:
            cards += str(card) + " "
        return cards.strip()


class Deck:
    '''Deck consists of a list of cards.
    This class is capable of dealing a card to a hand,
    shuffling a deck, and returning a printable string
    representation of a deck'''

    def __init__(self):
        '''deck refers to a list of cards.
        Since I am not accounting for suit,
        I add all of the cards 4 times to
        acheive the correct total'''
        self.deck = []
        for _ in range(4):
            for rank in RANKS:
                self.deck.append(Card(rank))

    def deal(self):
        '''Deal a single card and remove it
        from the deck.'''
        return self.deck.pop()

    def shuffle(self):
        # Shuffle the deck.
        random.shuffle(self.deck)

    def __str__(self):
        # Returns printable string representation of the deck.
        cards = ''
        for card in self.deck:
            cards += str(card) + " "
        return cards.strip()


def playerTurn(player, playerHand):
    '''This function takes in the player and the player's hand.
     This function facilitates the logic of a single turn
    in Blackjack'''
    global player1Stay, player2Stay

    # Print whose turn it is
    print('Turn: ' + player)

    # Ask the player if they want to view their hand and proceed accordingly
    playerHand.viewHand()

    # Ask the player if they want to hit or stay and proceed accordingly
    hitOrStay = input(player + ": Do you want to hit or stay? Type 'hit' to\
                      receive another card, type 'stay' to stop taking cards.\
                      \n")

    # User input check
    while hitOrStay.lower() != 'hit' and hitOrStay.lower() != 'stay':
        hitOrStay = input(player + ": Please type 'hit' or 'stay' to\
                          continue.\n")

    if hitOrStay.lower() == 'hit':
        # Deal a card and add it to the hand
        playerHand.add_card(deck.deal())
        print("\nOne card has been added to ", player, "'s hand.", sep="")
        # Print player's face up cards
        print(player + "'s face-up cards: ", end=' ', sep="")
        playerHand.print_face_up_cards()
        '''Ask the player if they want to view their hand and
        #proceed accordingly'''
        playerHand.viewHand()

    # Record if the player decides to stay
    elif hitOrStay.lower() == 'stay':
        if player == 'Player 1':
            player1Stay = True
        if player == 'Player 2':
            player2Stay = True
    return


def checkOutcome():
    # Check results of the game when both players are done
    global player1, player2
    # Print both player's hands and values
    print("Player 1 Hand: ", player1)
    print("Player 1 Hand Total: ", player1.get_value())
    print()
    print("Player 2 Hand: ", player2)
    print("Player 2 Hand Total: ", player2.get_value())

    # Print outcome if there is a tie or a winner
    if player1.get_value() == player2.get_value():
        print("Both players have ", player1.get_value(), "! There is a tie!",
              sep='')
    elif player1.get_value() == 21:
        print("Player 1 has Blackjack! Player 1 wins!")
    elif player2.get_value() == 21:
        print("Player 2 has Blackjack! Player 2 wins!")
    else:
        print("\n***Player 1 Wins!***" if player1.get_value() >
              player2.get_value() else "\n***Player 2 Wins!***")

# Global variables
deck = Deck()
player1Stay = False
player2Stay = False
playerBust = False
turn = 'Player 1'
player1 = Hand()
player2 = Hand()


def main():
    global player1Stay, player2Stay, playerBust, turn, deck, player1, player2

    # Shuffle the deck
    deck.shuffle()

    for _ in range(2):
        # Deal two cards to each player's hand
        player1.add_card(deck.deal())
        player2.add_card(deck.deal())
    print("\nThe cards have been dealt! Each player has two cards.")

    while (not player1Stay or not player2Stay) and not playerBust:
        '''While either player has not "stayed" and neither player's hand is
        over 21 (bust), game continues
        Print both player's face up cards'''
        print("Player 1's face-up cards: ", end='')
        player1.print_face_up_cards()
        print("Player 2's face-up cards: ", end='')
        player2.print_face_up_cards()
        print()

        # Facilitate the respective player's
        # turn and check if they have a bust
        if turn == 'Player 1':
            playerTurn(turn, player1)
            player1.bust()
        else:
            playerTurn(turn, player2)
            player2.bust()

        # Change turns
        if turn == 'Player 1' and not player2Stay:
            turn = 'Player 2'
        elif turn == 'Player 2' and not player1Stay:
            turn = 'Player 1'

    if not playerBust:
        # If both players have stayed, check the outcome to finish the game
        checkOutcome()

main()