import random
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    args = parser.parse_args()
    shuffle_file_lines(args.input_file, args.input_file + '_shuffle.txt')

def shuffle_file_lines(input_file, output_file, seed=123):
    """
    Reads a file, shuffles its lines, and writes the result to a new file.

    :param input_file: Path to the input file.
    :param output_file: Path to the output file.
    """
    try:
        random.seed(seed)
        # Read all lines from the input file
        with open(input_file, 'r') as file:
            lines = file.readlines()

        # Shuffle the lines randomly
        random.shuffle(lines)

        # Write the shuffled lines to the output file
        with open(output_file, 'w') as file:
            file.writelines(lines)

        print(f"Shuffled lines written to {output_file}")
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    main()