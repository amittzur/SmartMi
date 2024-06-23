def sum_dictionary_values(input_dict):
    """
    Sum all the values in a dictionary.

    Parameters:
        input_dict (dict): Input dictionary.

    Returns:
        float: Sum of all values.
    """
    return sum(input_dict.values())

# Example usage
my_dict = {'a': 10, 'b': 20, 'c': 30}

total_sum = sum_dictionary_values(my_dict)
print("Sum of dictionary values:", total_sum)
