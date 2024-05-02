import itertools

# Define the characters
characters = 'ABCD'

# Get all combinations of length 4
combinations = list(itertools.product(characters, repeat=4))

# Convert the combinations to strings
combinations = [''.join(combination) for combination in combinations]

# Print the combinations
for combination in combinations:
    print(combination)
with open('input.txt', 'w') as f:
    # Write the combinations to the file
    for combination in combinations:
        f.write(combination + '\n')