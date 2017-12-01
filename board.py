import numpy as np

class Square(object):
    """
    Represents one square of an othello board.

    Instance variables:
        value - If a disk is placed, its value, else None
        is_valid_move - boolean if the current player can play here
        flipped_disks - if the current player plays here,
                        which disks will flip to their color
    """

    def __init__(self, value=None):
        """
        Initialize Square.

        Optional arguments:
            value - current value of square, default None
        """
        self._value = value
        self.is_valid_move = False
        self.flipped_disks = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.is_valid_move = False
        self.flipped_disks = []


class Board(object):
    """
    8 x 8 game board which can be displayed and interacted with.

    Instance variables:
        board_state - an 8 x 8 numpy array of Square objects
        game_over - boolean: True if the game has ended
        current_player - 0/1: the player whose move it is
        p0_score - number of discs player 0 controls
        p1_score - number of discs player 1 controls
        verbose - boolean: print output on moves

    Methods:
        update_valid_moves - calculate valid moves for the current player
        get_valid_moves - return valid moves and flipped disks
        print_board - display board state in a human-readable format
        translate_move - return a coordinate move, (x, y) in
                         human-readable format
        human_move - given human-readable input (e.g. 'A4'), make move
        coord_move - given tuple (x, y), make move
    """
    _human_readable_dict = dict(enumerate('ABCDEFGH'))
    _computer_readable_dict = dict([(v, k) for k, v in enumerate('ABCDEFGH')])
    # _print_row = '%s %s %s %s %s %s %s %s'.encode('utf-8')
    print_row = '%s %s %s %s %s %s %s %s'

    def __init__(self, board_state=None, verbose=True):
        """
        Initialize Board object.

        Optional arguments:
            board_state - an 8 x 8 numpy array of Square objects
                          default: starting board
            verbose - print output; default True
        """
        self._current_player = 0
        self.verbose = verbose
        if board_state is None:
            start = np.array([[Square(1), Square(0)], [Square(0), Square(1)]])
            self.board_state = np.array([
                [Square() for _ in range(8)] for _ in range(8)])
            self.board_state[3:5, 3:5] = start
        else:
            self.board_state = board_state
        self.update_valid_moves()
        self.game_over = False
        self._score()
        self._xstr = np.vectorize(self._xstr)
        if self.verbose is True:
            self.print_board()

    def board_state_list(self):
        s = [[r.value if r.value is not None else -1 for r in row] for row in self.board_state]
        return np.array(s).reshape(1, 64)

    @property
    def current_player(self):
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        if value not in [0, 1]:
            raise ValueError('Player can only be 0 or 1.')
        self._current_player = value
        self.update_valid_moves()

    def get_valid_moves(self):
        """
        Return all valid moves and accompanying flipped disks.

        Returns:
            vms - a list of tuples where
                tuple[0] is a valid move and
                tuple[1] is a list of flipped disks
        """
        flipped_disks = np.array([
            [r.flipped_disks for r in row] for row in self.board_state])
        valid_moves = [tuple(r) for r in np.argwhere(flipped_disks)]
        flipped_moves = list(flipped_disks[np.where(flipped_disks)])
        return zip(valid_moves, flipped_moves)

    def _valid_moves_in_array(self, array):
        """
        Given a 1-D array, find valid moves and disks flipped.

        Called by update_valid_moves()

        Arguments:
            array - a 1-D numpy array

        Returns:
            valid_moves - list of tuples:
                tuple[0] is index of valid move
                tuple[1] is list of indices of flipped disks
        """
        valid_moves = []
        # If array doesn't have both 0 and 1, there are no valid moves.
        if not ((array == 0).any() and (array == 1).any()):
            return valid_moves
        opponent = 1 - self._current_player
        pos_list = np.where(array == self._current_player)[0]

        for pos in pos_list:
            i = pos - 1
            flipped_disks = []
            if i >= 0 and array[i] == opponent:
                flipped_disks += [i]
                i -= 1
                while i >= 0 and array[i] == opponent:
                    flipped_disks += [i]
                    i -= 1
                if i >= 0 and array[i] is None:
                    valid_moves += [(i, flipped_disks)]

            i = pos + 1
            flipped_disks = []
            if i < len(array) and array[i] == opponent:
                flipped_disks += [i]
                i += 1
                while i < len(array) and array[i] == opponent:
                    flipped_disks += [i]
                    i += 1
                if i < len(array) and array[i] is None:
                    valid_moves += [(i, flipped_disks)]
        return valid_moves

    def update_valid_moves(self):
        """
        Update valid moves for the current player, save to board_state.
        """
        # Reset all cells to 'invalid move'
        for row in self.board_state:
            for cell in row:
                cell.is_valid_move = False
                cell.flipped_disks = []

        # Horizontal
        for i, row in enumerate(self.board_state):
            array = np.array([cell.value for cell in row])
            vms = self._valid_moves_in_array(array)
            for index, flipped in vms:
                flipped_ids = zip([i] * len(flipped), flipped)
                self.board_state[i, index].is_valid_move = True
                self.board_state[i, index].flipped_disks += flipped_ids

        # Vertical
        for i, col in enumerate(self.board_state.T):
            array = np.array([cell.value for cell in col])
            vms = self._valid_moves_in_array(array)
            for index, flipped in vms:
                flipped_ids = zip(flipped, [i] * len(flipped))
                self.board_state[index, i].is_valid_move = True
                self.board_state[index, i].flipped_disks += flipped_ids

        # Diagonal NW--SE
        for i in range(-7, 8):
            array = np.array([
                cell.value for cell in self.board_state.diagonal(i)])
            vms = self._valid_moves_in_array(array)
            for index, flipped in vms:
                flipped_ids = [self._diag_coords(i, f, 'NW') for f in flipped]
                coords = self._diag_coords(i, index, 'NW')
                self.board_state[coords].is_valid_move = True
                self.board_state[coords].flipped_disks += flipped_ids

        # Diagonal NE--SW
        for i in range(-7, 8):
            array = np.array([cell.value for cell in np.diag(np.fliplr(self.board_state), i)])
            vms = self._valid_moves_in_array(array)
            for index, flipped in vms:
                flipped_ids = [self._diag_coords(i, f, 'NE') for f in flipped]
                coords = self._diag_coords(i, index, 'NE')
                self.board_state[coords].is_valid_move = True
                self.board_state[coords].flipped_disks += flipped_ids

    def _diag_coords(self, i, index, NW_NE):
        """
        Translate from diagonal indices to array coordinates.

        Called by update_valid_moves()

        Arguments:
            i - i'th slice of matrix
            index - index of the diagonal array
            NW_NE - can be 'NW' or 'NE'; direction of slice

        Returns:
            row - index of numpy row
            col - index of numpy column

        Exceptions Raised:
            AssertionError - if NW_NE is neither 'NW' nor 'NE'
        """
        assert NW_NE in ['NW', 'NE'], 'NW_NE must be either "NW" or "NE"'
        if i >= 0:
            row = index
        else:
            row = index - i

        if NW_NE == 'NW':
            col = row + i
        else:
            if i > 0:
                col = 7 - index - i
            else:
                col = 7 - index
        return row, col

    def print_board(self):
        """Display board state in a human-readable format."""
        row_num = 1
        for row in self.board_state:
            vals = tuple(self._xstr([cell for cell in row]))
            print(row_num, self.print_row % vals)
            row_num += 1
        print('  A B C D E F G H')


    def _xstr(self, s):
        """For printing human-readable board: convert None to ' '."""
        if s.value is None:
            if s.is_valid_move:
                return '+'
            else:
                return u'\u25E6'  # bullet
        elif s.value == 0:
            return u'\u25CF'  # filled circle
        elif s.value == 1:
            return u'\u25CB'  # empty circle
        else:
            raise ValueError('Invalid value for square.')


    def translate_move(self, move):
        """
        Translate coordinate move (x, y) into human-readable format.

        Arguments:
            move - an (x, y) tuple

        Returns:
            a length-2 string e.g. 'A4'
        """
        return self._human_readable_dict[move[1]], move[0] + 1


    def human_move(self, s):
        """
        Make move on board.

        Arguments:
            s - Human-readable string of length-2 e.g. 'A4'

        Effects:
            take move
            flip relevent disks
            update current_player
            update valid moves
            print message if player skipped
            check if game has ended; print final score if so
            print board

        Exceptions raised:
            AssertionError - if s is incorrect length
            RuntimeError - if invalid move provided
        """
        assert len(s) == 2, 's must be 2 characters'
        s1, s2 = list(s)
        s1 = self._computer_readable_dict[s1]
        s2 = int(s2) - 1
        self.coord_move((s2, s1))

    def coord_move(self, move):
        """
        Make move on board.

        Called by human_move()

        Arguments:
            move - tuple of array coordinates (x, y)

        Effects:
            take move
            flip relevent disks
            update current_player
            update valid moves
            print message if player skipped
            check if game has ended; print final score if so
            print board

        Exceptions raised:
            RuntimeError - if invalid move provided
        """

        cell = self.board_state[move]

        # print("move", move)
        if cell.is_valid_move is False:
            raise RuntimeError('Invalid move.')

        flipped_disks = cell.flipped_disks

        self.board_state[move].value = self._current_player
        for d in flipped_disks:
            self.board_state[d].value = self._current_player

        self._current_player = 1 - self._current_player
        self.update_valid_moves()
        self._score()
        if self.verbose:
            self.print_board()

    def _score(self):
        """
        Tracks if players have moves and if game has ended

        Called by coord_move()

        Effects:
            update current player scores
            if current player has no valid moves, skip player
            print 'skipped player' message
            update valid moves
            if current player has no valid moves
                set game_over to True
                print 'game over' message
                print final scores
        """
        vals = np.array([[r.value for r in row] for row in self.board_state])
        self.p0_score = len(np.where(vals == 0)[0])
        self.p1_score = len(np.where(vals == 1)[0])
        # n_moves = len(self.get_valid_moves())
        n_moves = len(list(self.get_valid_moves()))

        if n_moves == 0:
            self._current_player = 1 - self._current_player
            self.update_valid_moves()
            # n_moves = len(self.get_valid_moves())
            n_moves = len(list(self.get_valid_moves()))
            if n_moves == 0:
                self.game_over = True
                if self.verbose:
                    print('Game over.')
                    print(u'\u25CF : %s\t\u25CB : %s'.encode('utf-8') % (
                        self.p0_score, self.p1_score))
