from collections import defaultdict

DIR_UP = "u"
DIR_DOWN = "d"
DIR_LEFT = "l"
DIR_RIGHT = "r"
BLANK_PIECE = "Z"

def pretty_print(board):
    """Print the 2D-array board in a table format. Return None.
    
    Arguments: 
    board -- a 2D array containing pieces as upppercase characters
    """
    # Create the first row of indices
    first_line = "   "
    for i in range(len(board[0])):
        first_line += f"{i:<3d}"
    print(first_line)
    # Create the row of dashes
    print("   " + "-" * len(board[0]) * 3)
    # Create the rest of the rows of the board and 2 blank lines
    for i in range(len(board)):
        line = f"{i:>2d}" + "|"
        for j in range(len(board[0])):
            line += f"{board[i][j]:<3s}"
        print(line)
    print()

def validate_input(board, position, direction):
    """ Validate the inputs: the board, position and direction as specified
    in the specifications, return True if ALL is satisfied, False otherwise.
    
    Arguments: 
    board -- 2D array of uppercase char
    position -- 2-tuple of integers
    direction -- One of the 4 character "u", "d", "l", "r"
    """
    # Use 'lazy evaluation': 
    # Return False for the 1st occurence of invalid conditions

    # 1. Check if the board has at least 2 rows and 2 columns
    if len(board) < 2 or len(board[0]) < 2:
        return(False)
    
    # Check condition 2 and 3
    row_length = column_number = len(board[0])
    row_number = len(board)

    for row in board:
        # 2. Check all row has the same length
        if len(row) != row_length:
            return(False)
        # 3. Check all element is a character AND uppercase
        for board_value in row:
            if not(type(board_value) is str) or len(board_value) != 1 \
                   or not board_value.isupper():
                return(False)
    
    # 4. Check non-negative position within the board
    row_position, column_position = position
    if (row_position < 0 or row_position > (row_number - 1)):
        return(False)
    if (column_position < 0 or column_position > (column_number - 1)):
        return(False)
    
    # 5. Check direction in four permitted values
    if not (direction in (DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT)):
        return(False)
    
    # 6. Check the number of pieces is a multiple of four
    # Calculate the number of pieces for each colour
    colour_counts = {}
    for row in board:
        for board_value in row:
            if board_value != BLANK_PIECE:
                if board_value in colour_counts:
                    colour_counts[board_value] += 1
                else:
                    colour_counts[board_value] = 1

    # Check if the counts is divisible by 4
    for value in colour_counts.values():
        if value % 4 != 0:
            return(False)
    return(True)

def legal_move_no_change(board, position, direction):
    """Validate that a move is legal (both pieces are in the board, and at 
    least one ends in a position adjacent to the piece of the same colour) and 
    return True if legal, False otherwise.
    No change version: Perform no actual change in the board to boost 
    performance.
    
    Arguments: 
    board -- 2D array of uppercase char
    position -- 2-tuple of integers
    direction -- One of the 4 character "u", "d", "l", "r"
    """
    directions_dict = {DIR_UP: (-1, 0), DIR_DOWN: (1, 0), 
                      DIR_LEFT: (0, -1), DIR_RIGHT: (0, 1)}
    row_number = len(board)
    column_number = len(board[0])
    row_position, column_position = position
    new_row_position = row_position + directions_dict[direction][0]
    new_column_position = column_position + directions_dict[direction][1]

    # Check if both pieces are inside the board and not BLANK_PIECE
    if (new_row_position < 0 or new_row_position > (row_number - 1)):
        return(False)
    if (new_column_position < 0 or new_column_position > (column_number - 1)):
        return(False)
    if (row_position < 0 or row_position > (row_number - 1)):
        return(False)
    if (column_position < 0 or column_position > (column_number - 1)):
        return(False)
    if board[new_row_position][new_column_position] == BLANK_PIECE or \
        board[row_position][column_position] == BLANK_PIECE:
        return(False)
    
    # Check if at least one ends adjacent to another with same colour
    # by checking all 4 valid adjacent positions (if valid)
    # (row_curr, col_curr): The position of the current piece
    # (row_swap, col_swap): The position of the piece 
    # to be swapped into curr_position
    for (row_curr, col_curr, row_swap, col_swap) in \
        ((row_position, column_position, 
          new_row_position, new_column_position), 
         (new_row_position, new_column_position, 
          row_position, column_position)):
        for direction_vector in directions_dict.values():
            check_row = row_curr + direction_vector[0]
            check_col = col_curr + direction_vector[1]

            if 0 <= check_row < row_number:
                if 0 <= check_col < column_number:
                    # To avoid mutate the board, check the case when 
                    # the check position is the position to be swapped 
                    # (but not yet). Then, the actual piece at that check 
                    # position should be the piece at the current position.
                    if (row_swap, col_swap) == (check_row, check_col):
                        if board[row_curr][col_curr] == \
                            board[row_swap][col_swap]:
                            return(True)
                    elif board[row_swap][col_swap] == \
                        board[check_row][check_col]:
                        return(True)
    return(False)

def make_move(board, position, direction):
    """ Apply the move to the 2D array `board` and return the new board
    after performing all elimination and shifting.
    
    Arguments: 
    board -- 2D array of uppercase char
    position -- 2-tuple of integers
    direction -- One of the 4 character "u", "d", "l", "r"
    """
    # direction_dict: Store changes in position for each "direction"
    # new_row, new_column: Row and column position to be swapped
    directions_dict = {DIR_UP: (-1, 0), DIR_DOWN: (1, 0), 
                      DIR_LEFT: (0, -1), DIR_RIGHT: (0, 1)}
    row, column = position
    new_row = row + directions_dict[direction][0]
    new_column = column + directions_dict[direction][1]

    # Avoid mutation by creating a copy
    new_board = [row.copy() for row in board]

    # Swap the pieces
    temp = new_board[row][column]
    new_board[row][column] = new_board[new_row][new_column]
    new_board[new_row][new_column] = temp
    
    not_done = True  # flag for whether the process is done (no more removal)
    while not_done:
        not_done = False
        # Iterating from top to bottom, left to right for removing 2x2 box
        # so satisfy priorities: Lowest row first, then Lowest column first.
        max_row = len(new_board) - 1  # Max index for the row
        max_col = len(new_board[0]) - 1  # Max index for the column
        
        # When not_done = True (an elimination has been made), 
        # break from all FOR loop and restart.
        for i in range(max_row):
            if not_done:
                break
            for j in range(max_col):
                if not_done:
                    break
                # Check if 4 blocks in 2x2 region are equal and NOT blank
                if (new_board[i][j] != BLANK_PIECE) \
                    and (new_board[i][j] == new_board[i][j + 1]) \
                    and (new_board[i][j] == new_board[i + 1][j]) \
                    and (new_board[i + 1][j] == new_board[i + 1][j + 1]):
                    # Remove the region if valid and set flag to True
                    not_done = True
                    new_board[i][j] = new_board[i][j + 1] = BLANK_PIECE
                    new_board[i + 1][j] = new_board[i + 1][j + 1] = BLANK_PIECE

                    # Rearranging pieces after removal, sliding upward by 2
                    # move_i: The row position of the pieces to be shifted
                    # move_j: The column position of the pieces to be shifted
                    for move_i in range(i + 2, max_row + 1):
                        if new_board[move_i][j] != BLANK_PIECE:
                            new_board[move_i - 2][j] = new_board[move_i][j]
                            new_board[move_i][j] = BLANK_PIECE

                        if new_board[move_i][j + 1] != BLANK_PIECE:
                            new_board[move_i - 2][j + 1] = \
                                new_board[move_i][j + 1]
                            new_board[move_i][j + 1] = BLANK_PIECE
                    
                    # Rearranging pieces after removal, sliding leftward
                    for move_i in range(i, max_row + 1):
                        # Check if the left or rightmost piece is blank
                        # or no piece is blank, then shift appropriately
                        # shift: Store the position to be shifted
                        shift = 0

                        if new_board[move_i][j + 1] == BLANK_PIECE:
                            shift = 1
                            if new_board[move_i][j] == BLANK_PIECE:
                                shift = 2

                        if shift != 0:
                            for move_j in range(j + 2, max_col + 1):
                                new_board[move_i][move_j - shift] = \
                                    new_board[move_i][move_j]
                                new_board[move_i][move_j] = BLANK_PIECE
    return(new_board)

def list_2d_to_tuple(list_2d):
    """Convert 2d array to 2d tuple"""
    return(tuple([tuple(x) for x in list_2d]))

def calc_score(board):
    '''Calculate the score of board to rank the most likely move to empty table
    Score = Number of empty spaces - Sum of average distance 
    (for each letter group) between the first and other same letter;
    and the distance between each letter with its closest previous letter.
    The higher the score, the better.
    
    Arguments:
    board -- 2D array of uppercase characters
    '''

    # count_z: Count the value of empty space
    # first: Location of the first letter for each letter group
    # letter_distances: Euclidean distance between same letter to the first one
    # letter_counts: Count of each letter
    count_z = 0
    letter_distances = defaultdict(int)
    first = {}
    prev = {}
    letter_counts = defaultdict(int)

    for i in range(len(board)):
        for j in range(len(board[0])):
            letter = board[i][j]
            if letter == BLANK_PIECE:
                count_z += 1
            else:
                # Find Euclidean distances and
                # keep track of counts for average
                letter_counts[letter] += 1

                if letter in first:
                    first_i, first_j = first[letter]
                    letter_distances[letter] += \
                        ((i - first_i)**2 + (j - first_j)**2)**(0.5)
                else:
                    first[letter] = (i, j)

                if letter in prev:
                    prev_i, prev_j = first[letter]
                    letter_distances[letter] += \
                        ((i - prev_i)**2 + (j - prev_j)**2)**(0.5)
                    prev[letter] = (i, j)
                else:
                    prev[letter] = (i, j)

    total = 0
    for letter in letter_distances:
        # Find the total distances count: 2 * letter_counts[letter] - 3 = 
        # (letter_counts[letter] - 1) + (letter_counts[letter] - 2)
        total += letter_distances[letter] / (2 * letter_counts[letter] - 3)

    # Normalize the count_z and average distances
    max_avg_distance = ((len(board)**2 + len(board[0])**2)**0.5)
    normal_count_z = count_z / (len(board) * len(board[0]))
    normal_avg_distance = (total / len(first.keys())) / max_avg_distance

    return(normal_count_z - normal_avg_distance)

def binary_search(arr, num):
    """ Find the position of a score (num) in a descending sorted list of 
    list containing the score as the 3rd element
    
    Arguments:
    arr -- 2D list with each element containing the score as 3rd element
    num -- Float (the score of a board/ state)
    """
    
    if num >= arr[0][2]:
        return(0)
    elif num <= arr[-1][2]:
        return(len(arr))
    
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid][2] >= num:
            if arr[mid + 1][2] <= num:
                return(mid + 1)
            else:
                low = mid + 1
        elif arr[mid][2] < num:
            if arr[mid - 1][2] >= num:
                return(mid)
            else:
                high = mid - 1

def ai_player(board: list) -> list:
    """ Searching path to the empty board with best-first/ heuristic search-ish 
    approach. Return a series of moves to solve the board, or None if impossible
    to solve the board.
    
    Arguments:
    board -- 2D list of uppercase characters.
    """


    # directions: All 4 possible directions

    # queue: Store all the "partial" path up until a point
    # (let's call the return array a "path" to empty table) in the format 
    # [moved_board, partial path, score of move], sorted descendingly by score.
    # The first element is popped to calculate the improved "partial" path,
    # and each new "partial" path created is inserted into its correct place 
    # in the "queue" based on the score.

    # prev_state: Set of previous states of the board (as 2d tuple)
    # that has encountered. Used for eliminating looping move and 
    # ensure program terminates at some point.

    directions = [DIR_UP, DIR_DOWN, DIR_LEFT, DIR_RIGHT]
    queue = []
    prev_state = {list_2d_to_tuple(board)}

    # Initialisation of all possible move in the first scan of the board
    # Return if moved_board = empty, else add to the queue and prev_state
    if not validate_input(board, (0, 0), "r"):
        return(None)
    
    row_num = len(board)
    col_num = len(board[0])
    empty = [["Z"] * col_num for i in range(row_num)]  # Empty board 

    for i in range(row_num):
        for j in range(col_num):
            for direction in directions:
                if legal_move_no_change(board, (i, j), direction):
                    moved_board = make_move(board, (i, j), direction)

                    if moved_board == empty:
                        return([((i, j), direction)])
                    
                    tuple_form = list_2d_to_tuple(moved_board)
                    if tuple_form not in prev_state:
                        queue.append([moved_board, [((i, j), direction)], 
                                          calc_score(moved_board)])
                        prev_state.add(tuple_form)

    queue.sort(key=lambda x: x[2], reverse=True)
    # Continue scanning until found a solution (moved_board = empty), 
    # or no other move to change the board (queue is empty, when all states are
    # encountered)
    while len(queue) > 0:
        # check_queue: The queue to be checked next (exclude the first position
        # which is currently checked)
        new_board = queue[0][0]
        check_queue = queue[1:]

        for i in range(row_num):
            for j in range(col_num):
                for direction in directions:
                    if legal_move_no_change(new_board, (i, j), direction):
                        moved_board = make_move(new_board, (i, j), 
                                                    direction)
                        
                        if moved_board == empty:
                            queue[0][1].append(((i, j), direction))
                            return(queue[0][1])
                        
                        tuple_form = list_2d_to_tuple(moved_board)
                        if tuple_form not in prev_state:
                            # Calculate score, binary search the position of 
                            # the new entry in the sorted "queue", and insert
                            # it directly to avoid re-sorting
                            new_seq = queue[0][1] + [((i, j), direction)]
                            score = calc_score(moved_board)
                            position = binary_search(check_queue, score) + 1
                            queue.insert(position, [moved_board, new_seq, 
                                                        score])
                            prev_state.add(tuple_form)
        queue.pop(0)
    return(None)

# ------------------------------------------
# Testing on 4x4 matrix 
import json
import time

test_cases = []
with open("test_cases_4x4.json", "r") as fp:
    test_cases = json.load(fp)

min_time = -1
max_time = -1
total = 0
for board in test_cases[:100]:
    start = time.time()
    ai_player(board)
    end = time.time()

    time_required = end - start
    if (time_required > max_time):
        max_time = time_required

    if (min_time == -1):
        min_time = time_required
    elif (time_required < min_time):
        min_time = time_required
    
    total += time_required

print(f"Min time: {min_time}")
print(f"Max time: {max_time}")
print(f"Average time: {total / 100}")


# Test cases generation for 4x4 board:

# import json
# import random

# def valid_board(board):
#     # Check no 2x2 blocks of same character exists
#     for i in range(len(board) - 1):
#         for j in range(len(board[0]) - 1):
#             if (board[i][j] == board[i][j + 1]) \
#                 and (board[i][j] == board[i + 1][j]) \
#                 and (board[i + 1][j] == board[i + 1][j + 1]):
#                 return False
#     return True

# test_cases = []
# # 4 charaters
# for _ in range(5000):
#     count = {"A": 4, "B": 4, "C": 4, "D": 4}
#     choices = ["A", "B", "C", "D"]
#     board = [[""] * 4 for _ in range(4)]
#     for i in range(4):
#         for j in range(4):
#             # print(board)
#             cur_char = random.choice(choices)
#             board[i][j] = cur_char
#             count[cur_char] -= 1
#             if count[cur_char] == 0:
#                 choices.remove(cur_char)
#     if (valid_board(board) and board not in test_cases):
#         test_cases.append(board)

# # 2 characters
# for _ in range(5000):
#     count = {"A": 8, "B": 8}
#     choices = ["A", "B"]
#     board = [[""] * 4 for _ in range(4)]
#     for i in range(4):
#         for j in range(4):
#             # print(board)
#             cur_char = random.choice(choices)
#             board[i][j] = cur_char
#             count[cur_char] -= 1
#             if count[cur_char] == 0:
#                 choices.remove(cur_char)
#     if (valid_board(board) and board not in test_cases):
#         test_cases.append(board)

# with open("test_cases_4x4.json", "w") as file:
#     json.dump(test_cases, file)

