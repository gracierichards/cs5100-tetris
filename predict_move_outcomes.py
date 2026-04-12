import numpy as np

TETROMINOS = {
  "I": [
    [[1,1,1,1]],

    [[1],
     [1],
     [1],
     [1]]
  ],
  "O": [
    [[1, 1],
     [1, 1]]
  ],
  "T": [
    [[1,1,1],
     [0,1,0]],

    [[0,1],
     [1,1],
     [0,1]],

    [[0,1,0],
     [1,1,1]],

    [[1,0],
     [1,1],
     [1,0]],
  ],
  "L": [
    [[0, 0, 1],
     [1, 1, 1]],

    [[1, 0],
     [1, 0],
     [1, 1]],

    [[1, 1, 1],
     [1, 0, 0]],

    [[1, 1],
     [0, 1],
     [0, 1]]
  ],
  "J": [
    [[1, 0, 0],
     [1, 1, 1]],

    [[1, 1],
     [1, 0],
     [1, 0]],

    [[1, 1, 1],
     [0, 0, 1]],

    [[0, 1],
     [0, 1],
     [1, 1]]
  ],
  "S": [
    [[0, 1, 1],
     [1, 1, 0]],

    [[1, 0],
     [1, 1],
     [0, 1]]
  ],
  "Z": [
    [[1, 1, 0],
     [0, 1, 1]],

    [[0, 1],
     [1, 1],
     [1, 0]]
  ]
}

# Checks if the given tetromino (shape) is out of bounds or overlaps with any filled squares on the board
def collision(board, shape, row, col):
  h, w = shape.shape
  for r in range(h):
    for c in range(w):
      if shape[r][c]:
        br = row + r
        bc = col + c
        if br >= len(board) or bc >= len(board[0]):
          return True
        if board[br][bc]:
          return True
  return False

# Row at which the piece will stop moving if dropped at col
def landing_height(board, shape, col):
    row = 0
    while not collision(board, shape, row + 1, col):
        row += 1
    return row

# Returns a copy of board with the given piece filled in at the given position
def place(board, shape, row, col):
  new_board = board.copy()
  h, w = shape.shape
  for r in range(h):
    for c in range(w):
      if shape[r][c]:
        new_board[row + r][col + c] = 1
  return new_board

# Create a list of the resulting boards from each combination of rotation and drop column
def list_possibilities(board, piece_type):
  placements = []
  # iterate through the rotations for piece_type in TETROMINOS
  for i, shape in enumerate(TETROMINOS[piece_type]):
    shape = np.array(shape)
    width = shape.shape[1]
    # try every column that keeps the shape within bounds
    for col in range(10 - width + 1):
      if collision(board, shape, 0, col):  # collides even without dropping
        continue
      row = landing_height(board, shape, col)
      new_board = place(board, shape, row, col)
      placements.append({
        "rotation": i,
        "col": col,
        "board": new_board
      })
  return placements