# Ride the Bus: A Monte Carlo Analysis

A statistical look at the drinking game Ride the Bus, modeled as a Markov chain to measure expected drinks and variance across three player strategies.

Accompanies a Substack article: [link]

## The game

A player must answer four questions correctly in a row: Red or Black, Higher or Lower, Inside or Outside, and Suit. Any wrong guess is one drink and a restart to the first question. Cards are discarded as they are drawn, so the deck thins out over time. The goal is to drink as little as possible.

## The strategies

- **Random:** guesses that ignore the deck entirely.
- **Optimal:** perfect card counting, always playing the most likely option.
- **Last-5 human:** a realistic player who remembers only the last five cards.

## Key finding

Card counting barely helps, because the suit question is the great equalizer. Only three cards are gone before the suit guess, so with four suits one is always intact, and even a perfect counter can rarely beat a blind guess. The optimal player builds a big lead on the middle two questions and then loses most of it on the last one, which leaves the realistic human closer to random than to optimal.
