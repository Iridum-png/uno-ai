from unoaiinterface import (Deck, Discard, Player)

class Game:
    def __init__(self, player_count: int=4):
        self.deck = Deck()
        self.discard = Discard()
        self.top_card = self.deck.draw_card()
        while self.top_card.action:
            print("Redrawing card")
            print(self.top_card)
            self.deck.add_card(self.top_card)
            self.deck.shuffle()
            self.top_card = self.deck.draw_card()
        self.discard.add(self.top_card)
        self.players = [Player() for _ in range(player_count)]
        # self.deal()


    def deal(self) -> tuple:
        """
        Deals 7 cards to each player.

        Parameters:
            deck (Deck): The deck
            players (list): The list of players

        Returns:
            deck (Deck): The deck
            players (list): The list of players
        """
        for player in self.players:
            for _ in range(7):
                player.give(self.deck.draw_card())


    def turn(player: Player, discard: Discard, deck: Deck) -> int:
        """
        Handles the logic for a player's turn.

        Parameters:
            player (Player): The player whose turn it is
            discard (Discard): The discard pile
            deck (Deck): The deck

        Returns:
            bool: The outcome of the players turn
        """
        for card in player.hand:
            if card.is_playable(discard.top_card):
                break
        else:
            print("You have no playable cards. Drawing...")
            player.give(deck.draw_card())
            return 0
        print("Your hand: ")
        print(player)
        print(f"\nTop card: {discard.top_card}")

        card = input("Which card would you like to play? ")
        while not (card.isnumeric() and player.play_card(int(card)-1, discard)):
            print("That card is not playable.")
            card = input("Which card would you like to play? ")

        print(f"Playing card: {discard.top_card}")
        
        if len(player.hand) == 0:
            print("You win!")
            return -1
        elif len(player.hand) == 1:
            print("UNO!")
        elif action := discard.top_card.action:
            return {"skip": 1, "switch": 2, "two": 3, "wild": 4, "four": 5}[action]
        return 0


    # def run():
    #     """
    #     The main logic for running the game.
    #     """
    #     deck, discard, players = 
    #     deck, players = deal(deck, players)
    #     current_player = 0
    #     direction = 1

    #     while True:
    #         print(f"\nPlayer {current_player+1}'s turn")
    #         result = turn(players[current_player], discard, deck)
    #         match result:
    #             case -1:
    #                 break
    #             case 1:
    #                 print("Skipping next player's turn.")
    #                 current_player += 1
    #             case 2:
    #                 print("Reversing direction of play.")
    #                 direction *= -1
    #             case 3:
    #                 print("Next player draws 2!")
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 current_player += 1
    #             case 4:
    #                 print("Changing color to...")
    #                 color = input("Red, Blue, Green, or Yellow? ").title()
    #                 while color not in ["Red", "Blue", "Green", "Yellow"]:
    #                     print("Invalid color.")
    #                     color = input("Red, Blue, Green, or Yellow? ").title()
    #                 discard.top_card.color = color
    #             case 5:
    #                 print("Next player draws 4!")
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 players[(current_player + 1) % len(players)].give(deck.draw_card())
    #                 current_player += 1
    #                 print("Changing color to...")
    #                 color = input("Red, Blue, Green, or Yellow? ").title()
    #                 while color not in ["Red", "Blue", "Green", "Yellow"]:
    #                     print("Invalid color.")
    #                     color = input("Red, Blue, Green, or Yellow? ").title()
    #                 discard.top_card.color = color

    #         current_player = (current_player + 1) % len(players)

    #     print(f"Player {current_player+1} wins!")

print("Starting")
game = Game()
print(len(game.deck.cards))
