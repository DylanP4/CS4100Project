# Codenames — Desktop Application

A complete Python desktop implementation of the board game **Codenames** (classic edition) with a visual PyQt6 interface.

## Official Dataset Sources

- **Word list**: Official-style Codenames vocabulary (400 words).  
  - Source: [thomasahle/codenames — wordlist](https://github.com/thomasahle/codenames/blob/master/wordlist)  
  - A local copy is bundled in `data/words.txt` for offline play.

- **Key card distribution**: Generated programmatically to match the official rules:
  - **9** cards for the starting team  
  - **8** cards for the other team  
  - **7** neutral (bystander)  
  - **1** assassin  

  Reference: [Official Codenames rules](https://www.ultraboardgames.com/codenames/game-rules.php), [Czech Games](https://www.czechgames.com/games/codenames/).

## Requirements

- Python 3.10+
- PyQt6

## Setup

1. Clone or download this repository.
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

From the **project root** (`codenames/`):

```bash
python -m src.main
```

Or:

```bash
python src/main.py
```

From inside `src/`:

```bash
cd src && python main.py
```

## Rule Summary

- **4 players**: Red Spymaster, Red Operative, Blue Spymaster, Blue Operative.
- **Board**: 25 words in a 5×5 grid; each has a hidden type (red, blue, neutral, assassin).
- **Starting team**: The team with 9 words goes first.
- **Turn**:
  1. **Spymaster phase**: Spymaster gives one **word** clue and a **number** (≥ 1). The clue must not be any word on the board.
  2. **Operative phase**: Operative may make up to **(number + 1)** guesses and must make at least one.  
     - Correct team word → continue guessing.  
     - Neutral → turn ends.  
     - Opponent word → turn ends.  
     - Assassin → game ends; that team loses.
- **Win**: A team wins when all of its words are revealed.

## Features

- **Visual 5×5 board**: Unrevealed cards are gray; revealed cards show red, blue, beige (neutral), or black (assassin).
- **Status panel**: Current team, role (Spymaster/Operative), and words remaining per team.
- **Clue panel**: Spymaster enters clue word and number, then submits.
- **Guess panel**: Operative sees current clue and guesses remaining; can end turn early.
- **Spymaster view**: Optional toggle to show key colors on all cards (for the Spymaster’s eyes only).
- **Restart**: Use “Start Game” to begin a new game (available when no game is in progress or after a game ends).

## Project Layout

```
codenames/
├── data/
│   └── words.txt          # Bundled word list (fallback)
├── src/
│   ├── main.py            # Entry point
│   ├── game_engine.py     # Turn logic, clues, guesses, win/loss
│   ├── board.py           # Board state and reveal logic
│   ├── dataset_loader.py  # Word list and key assignment
│   └── ui/
│       ├── main_window.py # Main window and game flow
│       ├── board_widget.py# 5×5 word grid
│       ├── clue_panel.py  # Clue input and guess panel
│       └── status_panel.py# Team, role, remaining counts
├── requirements.txt
└── README.md
```

## License

This project is for educational and personal use. Codenames is a trademark of Czech Games Edition.
