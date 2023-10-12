############################################################################
# Ryan Hsu
# AITic - AI learning TIC TAC TOE
############################################################################
import numpy as np
import random

from tqdm import tqdm
import time

# Tic-Tac-Toe board size
BOARD_SIZE = 3

# Q-learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EXPLORATION_PROB = 0.3

# Initialize the Q-table
q_table = {}


# Initialize the Tic-Tac-Toe board
def initialize_board():
    """Initialize TIC TAC TOE board as a numpy array with all zeros"""
    return np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

# Define tic-tac-toe rules
def check_winner(board, player):
    """
    Checks if a player has won the game
    Parameters
    ------------------------------------
    board : a tic-tac-toe board that is a numpy 2d array of size 3 x 3
    player : player identification 1 or 2
    """
    # Check rows, columns, and diagonals
    for i in range(BOARD_SIZE):
        if all(board[i, :] == player) or all(board[:, i] == player):
            return True
    if all(np.diag(board) == player) or all(np.diag(np.fliplr(board)) == player):
        return True
    return False

def is_draw(board):
    """Checks if there are no more possible moves on the board"""
    return np.count_nonzero(board == 0) == 0 # if there are are no more empty spaces, it's a draw

def is_game_over(board):
    """Checks for end game state where either player or opponent is a winner or the game is a draw"""
    return check_winner(board, 1) or check_winner(board, 2) or is_draw(board)

def saving_move(board, player):
    """
    Checks if a player has saved themself from losing
    Parameters
    ------------------------------------
    board : a tic-tac-toe board that is a numpy 2d array of size 3 x 3
    player : player identification 1 or 2
    """
    if player == 1:
        opp = 2
    else:
        opp = 1

    # check all possible rows, columns and diags for a case where
    # the player has one piece and the opponent has two 
    for i in range(BOARD_SIZE):
        row = board[i,:].tolist()
        if row.count(player) == 1 & row.count(opp) == 2:
            return True
        
        col = board[:,i].tolist()
        if col.count(player) == 1 & col.count(opp) == 2:
            return True
        
        diag = np.diag(board).tolist()
        if diag.count(player) == 1 & diag.count(opp) == 2:
            return True       

        diag = np.diag(np.fliplr(board)).tolist() 
        if diag.count(player) == 1 & diag.count(opp) == 2:
            return True       
        
    return False

def about_to_lose(board,player):
    """
    Checks if a player is in a condition where they are about to lose
    because the opponent has two pieces and the player has non to block it
    Parameters
    ----------
    board : a tic-tac-toe board that is a numpy 2d array of size 3 x 3
    player : player identification 1 or 2
    
    Variables
    ---------
    count : integer that counts the occurences of each condition of losing
    """ 
    count = 0   
    if player == 1:
        opp = 2
    else:
        opp = 1

    for i in range(BOARD_SIZE):
        row = board[i,:].tolist()
        if row.count(opp) == 2 & row.count(player) == 0:
            count += 1
            #return True
        
        col = board[:,i].tolist()
        if col.count(opp) == 2 & row.count(player) == 0:
            count += 1
            #return True
        
        diag = np.diag(board).tolist()
        if diag.count(opp) == 2 & row.count(player) == 0:
            count += 1
            #return True       

        diag = np.diag(np.fliplr(board)).tolist() 
        if diag.count(opp) == 2 & row.count(player) == 0:
            count += 1
            #return True  
    #return False      
    return count
   
def get_available_actions(board):
    """Returns a list of available actions (empty cells) on the board"""
    actions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board[i][j] == 0:
                actions.append((i, j))
    return actions

def state_to_str(board):
    """Converts a board state to a string for use as a dictionary key"""
    return ''.join(str(cell) for row in board for cell in row)


def evaluate_state(board, previous_state):
    """
    Evaluate the board state and return a reward
    Parameters
    ----------
    board : current state of the board in play
    previous_state : the previous state of the board in play
    """
    reward = 0
    # if the AI has created a fork where there are two opportunities to win give pos reward
    if about_to_lose(board,1) >= 2:
        reward += 1
    # if the AI was about to lose but made a move that prevented the loss give pos reward
    if about_to_lose(previous_state, 1) >= 1  and saving_move(board,1):
        reward += 1
    # if the AI was about to win but didn't play the winning move give a neg reward
    if about_to_lose(previous_state,2) >= 1 and about_to_lose(board, 2) >= 1:
        reward -= 1
    # if AI move did not solve losing give a neg reward
    if about_to_lose(board,1) >= 1:
        reward -= 5   
    # if AI move cause the opponent to be about to lose (2 in a row) give pos reward
    if about_to_lose(board,2) >= 1:
        reward += 1    
    # check end-game state and give reward
    if check_winner(board, 1):
        return reward + 1  # Player 1 (X) wins
    elif check_winner(board, 2):
        return reward - 6  # Player 2 (O) wins
    else:
        return reward + 0  # It's a draw

def choose_action(state):
    """
    Return an action based on the epsilon-greedy policy
    Where either a random available move is taken or the best available move
    based on the current Q-table is taken
    """
    if random.uniform(0, 1) < EXPLORATION_PROB:
        # Exploration: Choose a random action
        return random.choice(get_available_actions(state))
    else:
        # Exploitation: Choose the action with the highest Q-value
        state_str = state_to_str(state)
        if state_str not in q_table:
            return random.choice(get_available_actions(state))
        
        max_move = "" 
        valid_action = False
            # Choose the action with the highest Q-value
        while not valid_action:
            actions = sorted(q_table[state_to_str(state)],key=q_table[state_to_str(state)].get, reverse=False)
            for act in actions:
                if act in get_available_actions(state):
                    max_move = act
                    valid_action = True
            
            if max_move == "":
                max_move = random.choice(get_available_actions(state))
                valid_action = True
            
        #max_move = max(q_table[state_str], key=q_table[state_str].get)
        return max_move

def update_q_value(state, action, reward, next_state):
    """
    Updates the Q-value using the Q-learning update rule
    Parameters
    ----------
    state : the current state of the board
    action : the action that will be taken
    reward : the reward to be given for the action at this state
    next_state : the next state with the move placed on the board
    """

    # convert state to a string representation to use as a dictionary key
    state_str = state_to_str(state)
    next_state_str = state_to_str(next_state)
    
    # add default entry into dictionary if state doesn't exist yet
    if state_str not in q_table:
        q_table[state_str] = {(0,0): 0.0}
    
    # add default entry for NEXT state if it doesn't exist yet
    if next_state_str not in q_table:
        q_table[next_state_str] = {(0,0): 0.0}
    
    # grab the max value of the next state if it exists, otherwise 0
    max_next_q_value = max(q_table[next_state_str].values(), default=0)
    
    # add the action (location of move) in board to the value dictionary
    if action not in q_table[state_str]:
        q_table[state_str][action] = 0
    
    # Bellman Optimality Equation
    q_table[state_str][action] = (1 - LEARNING_RATE) * q_table[state_str][action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_next_q_value)

def play_game():
    """
    Play a game of Tic-Tac-Toe pitting AI against itself
    And update Q-table while playing 
    """
    board = initialize_board()
    state = board.copy()
    player = 1  # Player 1 (AI) is '1', Player 2 (AI) is '2'

    # game loop until end game state reached
    while not is_game_over(board):
        action = choose_action(state)
        i, j = action
        board[i][j] = player

        # change player
        if player == 1:
            player = 2
        else:
            player = 1

        next_state = board.copy()
        # get the reward value for this action
        reward = evaluate_state(board, state)
        update_q_value(state, action, reward, next_state)
        state = next_state.copy()
    return board

# Train the AI by playing 10000  games
NUM_EPISODES = 10000
# print("playing game {} times....".format(NUM_EPISODES))
# for ep in range(NUM_EPISODES):
#     #print(ep)
#     play_game()
# print("Training Completed!")

for i in tqdm (range (NUM_EPISODES), 
               desc="Training by playing games…", 
               ascii=False, ncols=75):
    play_game()
print("Training Completed!")    

def play_against_ai():
    """
    Play a manual game against the trained AI
    """
    board = initialize_board()
    state = board.copy()
    while not is_game_over(board):
        if state_to_str(state) not in q_table:
            # If the state is not in the Q-table, choose a random action
            action = random.choice(get_available_actions(board))
        else:
            valid_action = False
            action = ""
            # Choose the action with the highest Q-value
            while not valid_action:
                actions = sorted(q_table[state_to_str(state)],key=q_table[state_to_str(state)].get, reverse=False)
                for act in actions:
                    if act in get_available_actions(board):

                        action = act
                        valid_action = True
                
                if action =="":
                    action = random.choice(get_available_actions(board))
                    valid_action = True

        i, j = action
        board[i][j] = 1  # AI is Player 1 (X)
        print("AI move: {}".format(action))
        print(board)
        if is_game_over(board):
            if is_draw(board) and check_winner(board,1) is False:
                print("Draw!")
            else:
                print("AI won!")
            break

        # Manual Turn
        valid_turn = False
        while not valid_turn:
            opp_action = int(input("Enter opponent's move (0-8): "))
            i, j = divmod(opp_action, BOARD_SIZE)

            if (i,j) in get_available_actions(board):
                valid_turn = True
            else:
                print("Not a valid move")
        
        board[i][j] = 2  # AI is Player 2 (O)
        print("Opponent move:")
        print(board)
        if is_game_over(board):
            if is_draw(board) and check_winner(board,2) is False:
                print("Draw!")
            else:
                print("Opponent won!")
            break

# asks to play a game
play = input("Want to play (y/n)?")
while play.upper() == 'Y':
    play_against_ai()
    play == 'N'
    play = input("Want to play (y/n)?")
